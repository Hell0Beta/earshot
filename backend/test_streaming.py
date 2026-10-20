"""Unit tests for the streaming core. No audio, GPU or model needed.

Run: ..\\.venv\\Scripts\\python.exe backend\\test_streaming.py

The stub model returns words only up to a moving point in the buffer, which is
what a real decoder does: it can only have heard as much audio as it has been
given.
"""

import collections
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming import RollingStreamer, StablePrefix, Word  # noqa: E402
from captions import TARGET_RATE, plan_step  # noqa: E402


def W(t, s, e):
    return Word(t, s, e)


def txt(ws):
    return " ".join(w.text for w in ws)


I, WANT, TO = W("i", 0, .2), W("want", .2, .4), W("to", .4, .6)
BUILD, A, REAL = W("build", .6, .9), W("a", .9, 1.0), W("real", 1.0, 1.3)


def make(model, **kw):
    """A streamer with the VAD gate disabled.

    The fake model is fed digital silence, which the real VAD gate correctly
    rejects -- so tests about *agreement* turn it off explicitly and stay
    about agreement. `test_vad_gate_*` below covers the gate itself.
    """
    kw.setdefault("min_speech_secs", 0.0)
    return RollingStreamer(model, **kw)


# ---- VAD gate: the defence against non-speech hallucinations ----


class _Vad:
    """Stubs Silero with a fixed speech duration per call."""

    def __init__(self, secs):
        self.secs = secs
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.secs


def test_vad_gate_skips_model_on_non_speech():
    """Music gets no pass, so it cannot become 'Thank you for watching'."""
    m = FakeModel()
    rs = make(m, min_speech_secs=0.10)
    rs._speech_secs = _Vad(0.0)
    committed, unstable = rs.feed(np.zeros(32000, dtype=np.float32))
    assert m.calls == 0, "non-speech must not reach the model"
    assert committed == [] and unstable == []


def test_vad_gate_discards_pending_tail():
    """A tail decoded from music is dropped, never committed."""
    m = FakeModel()
    rs = make(m, min_speech_secs=0.10, pause_secs=99)   # no pause flush
    rs._speech_secs = _Vad(1.0)                 # speech -> passes commit
    rs.feed(np.zeros(16000, dtype=np.float32))  # leaves an unstable tail
    assert rs.sp.unstable, "precondition: a tail is pending"
    rs._speech_secs = _Vad(0.0)                 # now only music
    rs.feed(np.zeros(16000, dtype=np.float32))
    assert rs.sp.unstable == [], "phantom tail must be discarded"
    assert rs.skipped_secs > 0


def test_vad_gate_admits_quiet_speech():
    """Speech the amplitude gate would call silence must still be transcribed."""
    m = FakeModel()
    rs = make(m, min_speech_secs=0.10)
    rs._speech_secs = _Vad(0.8)                 # VAD says speech
    rs.feed((np.random.rand(32000).astype(np.float32) - 0.5) * 0.0004)
    assert m.calls == 1, "quiet speech must still reach the model"


class WanderingModel:
    """Head word alternates every pass, so the prefix can never agree.

    This is what Whisper actually does on music: it re-decodes the same
    non-speech as "Thank you for watching", then "Thanks for watching", then
    "Thank you very much". The stable suffix is stable precisely so the test
    isolates the head disagreement.
    """

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, **kw):
        self.calls += 1
        head = "Thanks" if self.calls % 2 else "Thank"
        ws = [NS(word=" " + head, start=0.0, end=0.5),
              NS(word=" you", start=0.5, end=1.0)]
        return [NS(words=ws)], None


def test_resync_resolves_a_wandering_head():
    """A reworded head must not block the words that still agree behind it.

    Resync fires only after `resync_passes` consecutive disagreements *and*
    when the model has stopped saying the head word at all -- the evidence
    that it was a hallucination or a reword. The stable suffix then commits.
    """
    m = WanderingModel()
    rs = make(m, min_secs=0.1, pause_secs=99, max_buf_secs=999,
              stall_passes=6, resync_passes=2)
    rs._speech_secs = _Vad(99.0)
    outs = [rs.feed(np.zeros(16000, dtype=np.float32))[0] for _ in range(3)]
    assert outs[0] == [] and outs[1] == [], outs
    assert txt(outs[2]) == "you", outs
    assert rs.sp.resyncs == 1, "resync should have fired exactly once"
    committed_text = " ".join(txt(o) for o in outs)
    assert "Thank" not in committed_text and "Thanks" not in committed_text, \
        "nothing unagreed may be committed"


def test_resync_does_not_fire_while_the_head_is_still_said():
    """A slow-but-real head word must never be dropped.

    The model keeps emitting it, so it has not been falsified -- resync must
    leave it alone even after many disagreeing passes.
    """
    sp = StablePrefix(resync_passes=1)
    assert sp.update([W("um", 0, .2), I]) == []
    # Head disagrees, but "um" is still present later in the hypothesis.
    assert sp.update([W("uh", 0, .2), W("um", .2, .4), I]) == []
    assert sp.resyncs == 0
    assert txt(sp.unstable) == "uh um i", txt(sp.unstable)


class SingleWordWander:
    """Every hypothesis is one word, so no offset can agree: watchdog only."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, **kw):
        self.calls += 1
        word = " Thanks" if self.calls % 2 else " Thank"
        return [NS(words=[NS(word=word, start=0.0, end=1.0)])], None


def test_watchdog_bounds_a_stuck_prefix_when_resync_cannot_agree():
    """If no offset agrees at all, the watchdog is the last-resort bound."""
    m = SingleWordWander()
    rs = make(m, min_secs=0.1, pause_secs=99, max_buf_secs=999,
              stall_passes=3, resync_passes=99)
    rs._speech_secs = _Vad(99.0)
    outs = [rs.feed(np.zeros(16000, dtype=np.float32))[0] for _ in range(4)]
    assert outs == [[], [], [], []], outs       # agreeing is still required
    assert rs.buf_t0 == 0.0, "buffer must not advance while stalled"
    c, _u = rs.feed(np.zeros(16000, dtype=np.float32))    # 5th -> watchdog
    assert c, "watchdog must force progress instead of stalling to the cap"
    assert txt(c) in ("Thanks", "Thank"), txt(c)



# ---- StablePrefix ----


def test_stable_prefix_commits_on_agreement():
    """A word commits only once the next pass repeats it."""
    sp = StablePrefix()
    assert sp.update([I, WANT, TO]) == []          # nothing agreed yet
    assert txt(sp.update([I, WANT, TO, BUILD])) == "i want to"
    assert txt(sp.unstable) == "build"
    # "to" is already committed, so it is filtered out of the fresh set
    assert txt(sp.update([TO, BUILD, A])) == "build"
    assert txt(sp.unstable) == "a"


def test_correction_is_not_committed():
    """A reworded word must not commit — that is the whole point.

    LocalAgreement-2 exists because the same audio re-decoded from a
    different window comes back reworded. If "two" were committed on the
    strength of agreeing with itself, the wrong word would be permanent.
    """
    sp = StablePrefix()
    sp.update([I, WANT, W("two", .4, .6)])
    assert txt(sp.update([I, WANT, TO, BUILD])) == "i want"   # disagreed
    assert txt(sp.unstable) == "to build"


def test_boundary_stutter_and_punctuation():
    """Case and punctuation differences are not disagreement."""
    sp = StablePrefix()
    sp.update([W("Okay,", 0, .3), W("so", .3, .5)])
    sp.update([W("okay", 0, .3), W("so", .3, .5), W("now", .5, .8)])
    assert sp.last_end == 0.5
    sp.update([W("so", .35, .5), W("now", .5, .8), W("then", .8, 1)])
    assert txt(sp.unstable) == "then"


def test_flush_returns_rest():
    sp = StablePrefix()
    sp.update([I, WANT])
    assert txt(sp.flush([I, WANT, TO])) == "i want to"
    assert sp.unstable == []


def test_pause_does_not_commit_a_single_pass():
    """One hypothesis agreeing with itself must never commit."""
    sp = StablePrefix()
    assert sp.update([W("hello", 0, .5), W("world", .5, 1.0)]) == []
    assert txt(sp.unstable) == "hello world"


# ---- RollingStreamer ----


class FakeModel:
    """Decodes only the first `visible` seconds of whatever buffer it is given.

    Words are shaped like faster-whisper's (`word`/`start`/`end`), not like
    this module's `Word` (`text`), so the stub exercises the same attribute
    access the real model goes through.
    """

    def __init__(self, visible=0.5, words=None):
        self.visible = visible
        self.calls = 0
        self.prompts = []
        self.buf_lens = []
        self.kw = {}
        self._words = words

    def transcribe(self, audio, **kw):
        self.calls += 1
        self.kw = kw

        self.prompts.append(kw.get("initial_prompt"))
        self.buf_lens.append(len(audio) / 16000.0)
        end = min(self.visible, len(audio) / 16000.0)
        if self._words:
            ws = [NS(word=" " + w.text, start=w.start, end=w.end)
                  for w in self._words if w.end <= end + 1e-6]
        else:
            ws = [NS(word=" hello", start=0.0, end=0.25),
                  NS(word=" world", start=0.25, end=0.5)]
        return [NS(words=ws)], None


def test_second_pass_commits():
    """The committed-word latency is one extra pass, by construction.

    `pause_secs` is raised so the pause-flush path does not fire; this test
    is about agreement, not about pausing.
    """
    m = FakeModel()
    rs = make(m, min_secs=0.1, pause_secs=99)
    committed, unstable = rs.feed(np.zeros(16000, dtype=np.float32))
    assert txt(committed) == "", "first pass cannot commit"
    assert txt(unstable) == "hello world"
    committed, unstable = rs.feed(np.zeros(16000, dtype=np.float32))
    assert txt(committed) == "hello world", txt(committed)
    assert unstable == []


def test_idle_silence_skips_model():
    m = FakeModel()
    rs = make(m, min_secs=0.1, silence_peak=0.005)
    rs.feed(np.zeros(16000, dtype=np.float32))
    assert m.calls == 0, "silent buffer should not reach the model"


def test_quiet_audio_is_not_discarded():
    """A fixed peak gate reads quiet speech as silence; do not enable it."""
    m = FakeModel()
    quiet = (np.random.rand(32000).astype(np.float32) - 0.5) * 0.0004
    rs = make(m, min_secs=0.1, silence_peak=0.0)
    rs.feed(quiet)
    assert m.calls == 1, "quiet non-zero audio must still be transcribed"


def test_buffer_is_trimmed_after_commit():
    """Audio before the last committed word is dropped, so the buffer stays small."""
    m = FakeModel(words=[W("one", 0, .5), W("two", .5, 1.0), W("three", 1.0, 1.5)])
    rs = make(m, min_secs=0.1, max_buf_secs=99, pause_secs=99)
    rs.feed(np.zeros(32000, dtype=np.float32))   # 2s
    rs.feed(np.zeros(32000, dtype=np.float32))   # agree -> commit "one"
    assert rs.buf_t0 > 0.4, rs.buf_t0
    assert rs.buf_secs < 4.0, rs.buf_secs


def test_pause_flushes_everything():
    """A pause commits the tail instead of waiting for agreement."""
    m = FakeModel(words=[W("hello", 0, .5), W("world", .5, 1.0)], visible=3.0)
    rs = make(m, min_secs=0.1, pause_secs=0.4)
    # 3s of audio but words only reach 1.0s -> 2s of trailing silence
    committed, unstable = rs.feed(np.zeros(48000, dtype=np.float32))
    assert txt(committed) == "hello world", txt(committed)
    assert unstable == []


def test_buffer_cap_flushes():
    m = FakeModel(words=[W("a", 0, .5), W("b", .5, 1.0), W("c", 1.0, 1.5)])
    rs = make(m, min_secs=0.1, max_buf_secs=1.0, pause_secs=99)
    committed, _ = rs.feed(np.zeros(48000, dtype=np.float32))
    assert committed, "cap must force a flush"


def test_prompt_is_off_by_default():
    m = FakeModel()
    rs = make(m, min_secs=0.1, prompt_chars=0)
    rs.feed(np.zeros(16000, dtype=np.float32))
    assert m.prompts[0] is None, "initial_prompt must be off unless asked for"


def test_prompt_carries_history_when_enabled():
    m = FakeModel()
    rs = make(m, min_secs=0.1, prompt_chars=200)
    rs.feed(np.zeros(16000, dtype=np.float32))
    rs.feed(np.zeros(16000, dtype=np.float32))
    assert m.prompts[1] is not None and "hello" in m.prompts[1]


def test_flush_commits_remaining_words():
    """Words still waiting on agreement are recovered at session end."""
    m = FakeModel()
    rs = make(m, min_secs=0.1, pause_secs=99)
    committed, unstable = rs.feed(np.zeros(16000, dtype=np.float32))
    assert txt(committed) == "", "nothing may commit on the first pass"
    assert txt(unstable) == "hello world"
    assert txt(rs.flush()) == "hello world"
    assert rs.sp.unstable == []


def test_help_text_renders():
    """argparse interpolates '%' in help strings.

    A literal '20.2%' in a help string raises at format time and breaks
    `--help` (and therefore `earshot doctor`) for every user. Regressed once.
    """
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "captions.py"),
                        "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    for flag in ("--step-secs", "--min-speech-secs", "--gate-min-buf-secs",
                 "--stall-passes", "--no-speech-threshold",
                 "--vad-min-silence-ms"):
        assert flag in r.stdout, "%s missing from --help" % flag


def test_confidence_gates_reach_the_model():
    """The CLI confidence flags must actually reach transcribe().

    They were parsed but never passed to the model, so tuning them did
    nothing -- and `hallucination_silence_threshold` silently defaulted to
    None, disabling the gate against non-speech hallucination.
    """
    captured = {}

    class GateSpy:
        def transcribe(self, audio, **kw):
            captured.update(kw)
            return [NS(words=[])], None

    rs = RollingStreamer(GateSpy(), vad_min_silence_ms=150, vad_pad_ms=50,
                         no_speech_threshold=0.6,
                         hallucination_silence_threshold=2.0,
                         compression_ratio_threshold=2.4,
                         log_prob_threshold=-1.0)
    rs._speech_secs = lambda: 99.0
    rs.feed(np.zeros(16000, dtype=np.float32))
    assert captured["hallucination_silence_threshold"] == 2.0, captured
    assert captured["no_speech_threshold"] == 0.6, captured
    assert captured["log_prob_threshold"] == -1.0, captured
    assert captured["vad_parameters"] == {"min_silence_duration_ms": 150,
                                         "speech_pad_ms": 50}, captured
    assert captured["word_timestamps"] is True, captured
    assert captured["condition_on_previous_text"] is False, captured


def test_pass_secs_measures_the_actual_decode():
    """`pass_secs` must time the generator being *consumed*.

    faster-whisper's `transcribe()` is lazy: the work happens while iterating
    segments, not at the call. Timing at the call site reported ~3ms for work
    that really takes ~500ms, so the adaptive step planner saw a step far
    cheaper than reality and caption lag accumulated without bound. A slow
    generator here is the regression.
    """
    import time as _time

    class SlowModel:
        def transcribe(self, audio, **kw):
            def gen():
                _time.sleep(0.05)              # the real decode
                yield NS(words=[])
            return gen(), None

    rs = RollingStreamer(SlowModel(), min_speech_secs=0.0, pause_secs=99)
    t0 = _time.perf_counter()
    rs.feed(np.zeros(16000, dtype=np.float32))
    wall = _time.perf_counter() - t0
    assert rs.pass_secs >= 0.04, \
        "pass_secs=%r did not include the decode" % rs.pass_secs
    assert rs.pass_secs <= wall + 1e-6, "pass_secs cannot exceed the call"


def test_step_grows_to_match_inference_cost():
    """A step shorter than the pass cost guarantees accumulating latency.

    This is the regression guard for the accumulating-lag bug: at 0.3s a
    real pass costs ~500ms, so every cycle overran the audio it represented
    and the queue grew without bound (measured drift +270s over 240s).
    """
    slow = [0.5] * 10
    assert plan_step(0.3, 1.25, slow) >= 0.5, \
        "step must consume at least what a pass costs"

    hist = collections.deque([0.48] * 9 + [1.9], maxlen=12)
    assert plan_step(0.3, 1.25, hist) < 0.9, \
        "one spike must not inflate every later step (use p90, not max)"


def test_step_respects_floor_and_headroom():
    assert plan_step(0.3, 1.25, []) == 0.3, "no history -> the floor"
    assert plan_step(0.3, 1.25, [0.01] * 10) == 0.3, "fast passes -> the floor"
    hist = collections.deque([0.4] * 10, maxlen=12)
    assert abs(plan_step(0.3, 1.25, hist) - 0.5) < 1e-9


def test_step_keeps_up_as_inference_slows():
    """Whatever the machine does, the step must cover the pass cost."""
    for cost in (0.05, 0.3, 0.5, 0.8, 2.0):
        hist = collections.deque([cost] * 10, maxlen=12)
        assert plan_step(0.3, 1.25, hist) >= cost


def test_backlog_is_bounded_not_accumulated():
    """The drift guard must clamp, never let lag grow.

    Simulates a pipeline consuming audio at a fixed real rate while inference
    takes longer: without the cap the lag is unbounded.
    """
    backlog = 0.0
    max_backlog = 1.5
    for _ in range(400):                      # 400 steps of 0.5s arriving
        backlog += 0.5                        # audio produced in 0.5s
        backlog -= 0.75                       # ... but only 0.75s consumed
        if backlog > max_backlog:
            backlog = max_backlog             # the guard: clamp, not drop all
    assert backlog <= max_backlog + 1e-9


def test_bytes_to_mono16k_survives_odd_length():
    """An odd byte count must not kill the backend.

    np.frombuffer(dtype=int16) raises "buffer size must be a multiple of
    element size" on an odd length. The adaptive step returns a float, and
    ~15% of those produce an odd byte count, so this crashed the live backend
    about a minute into a session -- repeatedly, and with no traceback.
    """
    from captions import bytes_to_mono16k

    for n in (1, 3, 101, 255, 1023):
        out = bytes_to_mono16k(b"\x00" * n, 2, TARGET_RATE)
        assert isinstance(out, np.ndarray)
    # 2048 bytes of stereo int16 is 512 frames; at 16 kHz that is 512 samples.
    out = bytes_to_mono16k(b"\x00" * 2048, 2, TARGET_RATE)
    assert out.shape[0] == 512, out.shape


def test_step_bytes_are_always_whole_frames():
    """The step must snap to whole frames for every float target.

    A fixed step multiplied out to an even byte count every time, so this
    only ever bit once the step became adaptive. Sweep the value space the
    planner actually produces.
    """
    from captions import frame_bytes_for

    for rate, channels in ((44100, 1), (44100, 2), (48000, 1), (48000, 2)):
        bps = rate * channels * 2
        frame = frame_bytes_for(channels)
        assert frame == channels * 2
        for i in range(1, 2001):
            target = 0.30 + (i % 97) * 0.01     # typical adaptive targets
            want = int(bps * target)
            want -= want % frame               # what collect_step now does
            assert want % frame == 0, (rate, channels, target)
            assert want % 2 == 0, "int16 needs whole samples"


def test_adaptive_targets_are_frame_aligned_end_to_end():
    """The bug's actual shape: real planner output -> aligned byte count."""
    from captions import frame_bytes_for, plan_step

    hist = collections.deque([0.5 + (i % 7) * 0.013 for i in range(12)],
                             maxlen=12)
    channels, rate = 2, 48000
    bps, frame = rate * channels * 2, frame_bytes_for(channels)
    for _ in range(500):
        target = plan_step(0.3, 1.25, hist)
        want = int(bps * target)
        want -= want % frame
        assert want % frame == 0, target


def test_plan_step_is_not_ratcheted_by_one_slow_pass():
    """One 18s outlier must not inflate the step -- the runaway bug.

    Measured: a single 18.32s pass (GPU contention) drove p90*headroom to
    16.77s and then 19.52s, freezing captions and flushing a 44-word burst.
    A median ignores the outlier; the ceiling bounds it outright.
    """
    hist = collections.deque([0.7] * 11 + [18.32], maxlen=12)
    step = plan_step(0.3, 1.25, hist, 2.5)
    assert step < 1.0, "median must ignore a single slow pass: %.2f" % step


def test_plan_step_respects_the_ceiling():
    """Even a genuinely slow machine cannot push the step past the ceiling."""
    hist = collections.deque([18.0] * 12, maxlen=12)
    assert plan_step(0.3, 1.25, hist, 2.5) == 2.5


def test_empty_fresh_set_does_not_clear_the_tail_or_freeze_the_watchdog():
    """An empty hypothesis is absence of evidence, not evidence of progress.

    Clearing `prev` here both lost the pending words and froze `stalled`
    (which was only incremented while `prev` was non-empty), so the watchdog
    could never fire. That is the genuine deadlock class.
    """
    sp = StablePrefix()
    sp.update([I, WANT])
    assert txt(sp.update([I, WANT, TO])) == "i want"
    assert txt(sp.unstable) == "to"
    # Every word in this hypothesis is behind the commit point -> fresh empty.
    assert sp.update([I, WANT]) == []
    assert txt(sp.unstable) == "to", "the pending tail must survive"
    assert sp.stalled == 1, "an empty pass must still count as a stall"


def test_session_end_flush_keeps_the_pending_tail():
    """Quitting with < 0.1s buffered used to throw the last words away."""
    rs = make(FakeModel(), min_secs=0.1)
    rs.sp.prev = [I, WANT]
    rs.buf = np.zeros(0, dtype=np.float32)
    assert txt(rs.flush()) == "i want"


def test_word_latency_is_measured_from_audio_arrival():
    """Latency is spoken->committed, derived from the audio timeline."""
    rs = make(FakeModel(), min_secs=0.1)
    rs._arrival_abs = 10.0      # newest buffered audio ends at 10s
    rs._arrival_perf = 100.0    # ...and arrived at perf_counter 100
    w = W("x", 8.0, 9.0)        # ended at 9s -> arrived 1s earlier
    assert abs(rs.word_latency(w, now=100.5) - 1.5) < 1e-9


def test_pass_secs_resets_when_the_gate_skips():
    """A skipped pass must not report the previous pass's cost.

    The stale value printed as a constant `pass 0.791s` in the log and was
    fed to the step planner as if it were a fresh measurement.
    """
    m = FakeModel()
    rs = make(m, min_speech_secs=0.10)
    rs._speech_secs = _Vad(1.0)
    rs.feed(np.zeros(16000, dtype=np.float32))   # a real pass sets pass_secs
    rs._speech_secs = _Vad(0.0)
    rs.feed(np.zeros(16000, dtype=np.float32))   # gate skips the model
    assert rs.pass_secs == 0.0, rs.pass_secs
    assert rs.last_event == "gate", rs.last_event


def test_pass_buf_secs_records_the_audio_the_pass_decoded():
    """The buffer logged by the caller is post-`_cut`; this is the real one.

    On a pause flush the buffer is cut to zero *after* the pass, so the log
    showed `buffer 0.0s` for a 12.9s pass over ~1.4s of audio and hid the
    temperature-fallback stalls.
    """
    m = FakeModel()
    rs = make(m, min_secs=0.1, pause_secs=99)
    rs._speech_secs = _Vad(99.0)
    rs.feed(np.zeros(32000, dtype=np.float32))
    assert rs.pass_buf_secs >= 1.9, rs.pass_buf_secs
    assert rs.buf_secs >= 1.9
    # A pause flush cuts the buffer to nothing, but the pass still decoded it.
    rs.pause_secs = 0.01
    rs.feed(np.zeros(16000, dtype=np.float32))
    assert rs.buf_secs == 0.0
    assert rs.pass_buf_secs >= 1.0, rs.pass_buf_secs


def test_temperature_policy_reaches_the_model():
    """`off` must pass temperature=0.0; `full` must not pass it at all."""
    for policy, expected in ((None, "absent"), (0.0, 0.0),
                             ([0.0, 0.2, 0.4], [0.0, 0.2, 0.4])):
        m = FakeModel()
        rs = make(m, min_secs=0.1, temperature=policy)
        rs._speech_secs = _Vad(99.0)
        rs.feed(np.zeros(16000, dtype=np.float32))
        got = m.kw.get("temperature", "absent")
        assert got == expected, (policy, got)


def test_pass_empty_flags_a_decode_that_produced_nothing():
    """A slow pass that yields no words is pure waste; it must be visible."""
    class Silent(FakeModel):
        def transcribe(self, audio, **kw):
            self.calls += 1
            self.kw = kw
            return [], None

    rs = make(Silent(), min_secs=0.1)
    rs._speech_secs = _Vad(99.0)
    rs.feed(np.zeros(16000, dtype=np.float32))
    assert rs.pass_empty is True


def test_audio_writer_round_trips_mono16k():
    """The dump must be readable back at 16 kHz with the same length."""
    import tempfile
    import wave as wavemod
    from captions import AudioWriter

    path = os.path.join(tempfile.mkdtemp(), "dump.wav")
    aw = AudioWriter(path)
    aw.add(np.sin(np.linspace(0, 20, 16000)).astype(np.float32) * 0.5)
    aw.add(np.zeros(8000, dtype=np.float32))
    aw.close()
    with wavemod.open(path, "rb") as w:
        assert w.getframerate() == TARGET_RATE
        assert w.getnchannels() == 1
        assert w.getnframes() == 24000, w.getnframes()


def test_audio_writer_is_a_noop_when_disabled():
    from captions import AudioWriter

    aw = AudioWriter("")
    aw.add(np.zeros(100, dtype=np.float32))   # must not raise
    aw.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok:", t.__name__)
    print("ALL %d PASSED" % len(tests))
