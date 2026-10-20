"""Rolling-buffer streaming with stable-prefix commits.

Replaces "transcribe each fixed window independently and append" with:

  1. keep ONE audio buffer since the last commit point
  2. every `--step-secs`, re-transcribe the whole buffer with word timestamps
  3. commit the words that two consecutive hypotheses agree on
     (LocalAgreement-2, the approach used by whisper_streaming)
  4. the rest is the "unstable" tail: shown in the live line, free to change
  5. a pause (no words for `pause_secs` at the end of the buffer) or a
     `max_buf_secs` overflow flushes everything

Why this replaced four earlier attempts at deduplication: every one of them
tried to detect overlapping audio by *comparing text across passes*, and
Whisper re-decodes the same audio differently from every window boundary, so
the restatement is never verbatim. Nothing asks that question here — agreement
between consecutive hypotheses over the same buffer decides what is new. See
docs/streaming-latency.md.

Whisper therefore always sees real context, so there are no short-window
hallucinations, and the live line self-corrects because it is replaced rather
than appended.
"""

import re
import time
from dataclasses import dataclass

import numpy as np

SR = 16000


@dataclass
class Word:
    """One decoded word with its absolute session timing."""

    text: str
    start: float  # absolute session seconds
    end: float


def _norm(s):
    """Compare words ignoring case and punctuation."""
    return re.sub(r"[^\w']", "", s.lower())


class StablePrefix:
    """Commit the words two consecutive hypotheses agree on.

    A word is committed only once the *next* pass repeats it in the same
    position. That costs one extra pass of latency but removes the guesswork:
    there is no similarity threshold, no anchor window, and no edit-distance
    budget to tune, because the decision is "did the model say this again?".
    """

    def __init__(self, tolerance=0.15, resync_passes=3, max_resync=2):
        self.tol = tolerance
        self.prev = []          # uncommitted words from the previous hypothesis
        self.last_end = 0.0     # absolute end of the last committed word
        self.last_norm = ""
        # Consecutive passes that committed nothing while a tail was pending.
        # A prefix match that never agrees blocks all output, so this is the
        # watchdog signal that guarantees forward progress.
        self.stalled = 0
        # Consecutive passes where the head word disagreed. Once this reaches
        # `resync_passes`, the head is treated as falsified and dropped (see
        # `update`). Distinct from `stalled`: `stalled` drives the watchdog,
        # which force-commits; resync *discards* the head and keeps committing
        # only words that still agree behind it.
        self.disagree = 0
        self.resync_passes = resync_passes
        self.max_resync = max_resync
        self.resyncs = 0        # total resyncs, for diagnostics

    def _fresh(self, hyp):
        """Drop words we have already committed, and any boundary restatement.

        Two filters. The first discards anything starting before the last
        commit (minus tolerance, since timings jitter at the boundary). The
        second drops a leading word that merely restates the last committed
        one -- the boundary stutter a fresh decode produces.
        """
        out = [w for w in hyp
               if _norm(w.text) and w.start >= self.last_end - self.tol]
        if out and self.last_norm and _norm(out[0].text) == self.last_norm \
                and out[0].start < self.last_end + self.tol:
            out = out[1:]
        return out

    def _commit(self, words):
        if words:
            self.last_end = words[-1].end
            self.last_norm = _norm(words[-1].text)
        return words

    @staticmethod
    def _lcp(a, b):
        n = 0
        for x, y in zip(a, b):
            if _norm(x.text) != _norm(y.text):
                break
            n += 1
        return n

    def _resync(self, new):
        """Agree at a bounded offset after the head has been falsified.

        `prev` and `new` are positionally aligned, so a single reworded head
        word makes every position behind it disagree. Comparing `prev[d:]` with
        `new[d:]` recovers the words that *do* still agree once the stale head
        is skipped. Only offsets up to `max_resync` are tried: an unbounded
        search would happily align unrelated text.

        Returns (offset, agreed_count), or None.
        """
        limit = min(self.max_resync, len(self.prev), len(new))
        best = None
        for d in range(1, limit + 1):
            cnt = self._lcp(self.prev[d:], new[d:])
            if cnt and (best is None or cnt > best[1]):
                best = (d, cnt)
        return best

    def update(self, hyp):
        """`hyp` is the full hypothesis for the current buffer.

        Returns the newly committed words; `unstable` holds the rest.
        """
        new = self._fresh(hyp)

        # An empty hypothesis is not evidence that the pending tail is wrong;
        # it is the absence of evidence. Clearing `prev` here silently dropped
        # uncommitted words *and* froze the watchdog, because `stalled` was only
        # incremented while `prev` was non-empty.
        if not new:
            if self.prev:
                self.stalled += 1
                self.disagree += 1
            return []

        n = self._lcp(self.prev, new)
        if n:
            self.prev = new[n:]
            self.stalled = 0
            self.disagree = 0
            return self._commit(new[:n])

        # The first hypothesis after a commit has nothing to agree with, so it
        # is not a stall and must not count toward the resync threshold.
        if not self.prev:
            self.prev = new
            return []

        self.stalled += 1
        self.disagree += 1

        # Resync: the head has disagreed for `resync_passes` passes. Drop it if
        # the model has stopped emitting it *anywhere* -- that is the evidence
        # that it was a hallucination or a reword, not a slow-but-real word.
        # Words that still agree behind the dropped head are committed
        # normally; nothing unagreed is ever committed.
        if self.prev and self.disagree >= self.resync_passes:
            head = _norm(self.prev[0].text)
            still_said = any(_norm(w.text) == head for w in new)
            found = None if still_said else self._resync(new)
            if found:
                d, cnt = found
                self.resyncs += 1
                self.disagree = 0
                self.stalled = 0
                self.prev = new[d + cnt:]
                return self._commit(new[d:d + cnt])

        self.prev = new
        return []

    def take_pending(self):
        """Hand back the uncommitted tail without decoding anything.

        Used when the session ends with less than a step of audio left: there
        is nothing to re-decode, but the pending words are real and must not be
        thrown away (they are the "last few words" that used to go missing).
        """
        out = self.prev
        self.prev = []
        self.stalled = 0
        self.disagree = 0
        return out

    def discard(self):
        """Drop the unstable tail without committing it.

        Used when the buffered audio turns out to contain no speech: the tail
        was decoded from music or room tone, so it is a hallucination and must
        not reach the transcript. Committing it (the pause/flush path) is
        exactly how "Thank you for watching" becomes permanent.
        """
        self.prev = []
        self.stalled = 0
        self.disagree = 0

    def flush(self, hyp):
        """Commit everything left (pause, buffer overflow, or shutdown)."""
        new = self._fresh(hyp)
        self.prev = []
        self.stalled = 0
        self.disagree = 0
        return self._commit(new)

    @property
    def unstable(self):
        return self.prev


class RollingStreamer:
    def __init__(self, model, language="en", max_buf_secs=12.0,
                 pause_secs=0.4, min_secs=0.5, prompt_chars=0,
                 beam_size=5, silence_peak=0.0, min_prob=0.0,
                 min_speech_secs=0.10, stall_passes=6,
                 gate_min_buf_secs=0.9, resync_passes=3,
                 vad_min_silence_ms=150, vad_pad_ms=50,
                 no_speech_threshold=0.6, hallucination_silence_threshold=2.0,
                 compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                 temperature=None):
        self.model = model
        self.language = language
        self.max_buf_secs = max_buf_secs
        self.pause_secs = pause_secs
        self.min_secs = min_secs
        # 0 disables the prompt. It carries vocabulary across a flush but is
        # the same hallucination risk as `condition_on_previous_text=True`,
        # which this pipeline deliberately keeps False.
        self.prompt_chars = prompt_chars
        self.beam_size = beam_size
        self.silence_peak = silence_peak
        self.min_prob = min_prob
        self.min_speech_secs = min_speech_secs
        self.gate_min_buf_secs = gate_min_buf_secs
        self.stall_passes = stall_passes
        self.resync_passes = resync_passes
        self.vad_min_silence_ms = vad_min_silence_ms
        self.vad_pad_ms = vad_pad_ms
        self.no_speech_threshold = no_speech_threshold
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.temperature = temperature
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_t0 = 0.0            # absolute time of buf[0]
        self.sp = StablePrefix(resync_passes=resync_passes)
        self.history = ""            # committed text, optional prompt
        self.pass_secs = 0.0         # last pass cost, for logging
        self.pass_buf_secs = 0.0     # buffer length *the last pass decoded*
        self.pass_empty = False      # last pass produced no words
        self.skipped_secs = 0.0      # non-speech audio skipped by the VAD gate
        self.last_event = ""         # what the last feed() did, for the log
        self._vad = None
        # Wall-clock anchor for latency: when the newest buffered audio arrived
        # and how far that audio extends in absolute session time. Audio arrives
        # in real time, so a word ending at `e` arrived at
        #   _arrival_perf - (_arrival_abs - e)
        # and its latency is the difference from when it is committed.
        self._arrival_perf = None
        self._arrival_abs = 0.0

    @property
    def buf_secs(self):
        return len(self.buf) / SR

    def word_latency(self, word, now=None):
        """Seconds between `word` finishing in the audio and being committed.

        This is the number that matters to a viewer: how stale the caption is
        relative to the speech. It is immune to the generator gotcha that
        plagued `pass_secs` (see `_transcribe`) because it is derived from the
        audio timeline, not from a timed call.
        """
        if self._arrival_perf is None:
            return 0.0
        now = time.perf_counter() if now is None else now
        arrived = self._arrival_perf - (self._arrival_abs - word.end)
        return max(0.0, now - arrived)

    def _speech_secs(self):
        """Seconds of the buffer that Silero VAD considers speech.

        This is the gate that stops hallucinations. Whisper, handed a window
        of music or room tone, emits stock non-speech phrases ("Thank you for
        watching", "Thanks for watching", "Bye"). Measured on a real capture:
        a 15s musical stretch yields 0.0s of VAD speech, while real narration
        yields ~85% of the buffer. The margin is wide enough to skip the model
        entirely on non-speech, and amplitude cannot make that call safely --
        quiet compressed speech is quiet too.
        """
        if self._vad is None:
            from faster_whisper.vad import VadOptions
            self._vad = VadOptions(
                min_silence_duration_ms=self.vad_min_silence_ms,
                speech_pad_ms=self.vad_pad_ms,
            )
        from faster_whisper.vad import get_speech_timestamps
        spans = get_speech_timestamps(self.buf, self._vad)
        return sum(d["end"] - d["start"] for d in spans) / SR

    def _transcribe(self):
        import time

        t0 = time.time()
        # Recorded *before* the decode: this is the audio the pass actually
        # sees. The buffer size logged by the caller is measured after `_cut`,
        # so on a flush it reads 0.0s -- which made a 12.9s pass over 1.4s of
        # audio look like a pass over nothing, and hid the temperature-fallback
        # stalls for weeks.
        self.pass_buf_secs = self.buf_secs
        kwargs = {}
        if self.temperature is not None:
            # Passing a float disables faster-whisper's temperature fallback;
            # a list caps how many retries a failing segment may burn.
            kwargs["temperature"] = self.temperature
        segs, _ = self.model.transcribe(
            self.buf,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=self.vad_min_silence_ms,
                speech_pad_ms=self.vad_pad_ms,
            ),
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=(self.history[-self.prompt_chars:]
                            if self.prompt_chars else None),
            no_speech_threshold=self.no_speech_threshold,
            hallucination_silence_threshold=self.hallucination_silence_threshold,
            compression_ratio_threshold=self.compression_ratio_threshold,
            log_prob_threshold=self.log_prob_threshold,
            **kwargs,
        )
        out = []
        for s in segs:
            for w in (s.words or []):
                # Words the model itself is unsure of are never shown and
                # never committed, whatever the pause/flush path decides.
                if self.min_prob and w.probability < self.min_prob:
                    continue
                out.append(Word(w.word.strip(),
                                self.buf_t0 + w.start,
                                self.buf_t0 + w.end))
        # Timed *after* the loop, deliberately. `transcribe()` returns a lazy
        # generator: the decoding happens while iterating `segs`, not at the
        # call. Measuring at the call site reported ~3ms for work that really
        # takes ~500ms, which fed the adaptive step planner a meaningless
        # number and let caption lag accumulate without bound.
        self.pass_secs = time.time() - t0
        self.pass_empty = not out
        return out

    def _cut(self, t):
        """Drop buffered audio before absolute time `t`.

        Only audio behind the last *committed* word is dropped. The unstable
        tail stays buffered so it can be re-decoded next pass, and so
        `flush()` can recover it.
        """
        n = int(max(t - self.buf_t0, 0.0) * SR)
        n = min(n, len(self.buf))
        self.buf = self.buf[n:]
        self.buf_t0 += n / SR

    def feed(self, audio):
        """Add `audio` (float32 mono 16 kHz) and return (committed, unstable)."""
        self.buf = np.concatenate([self.buf, audio.astype(np.float32)])
        self._arrival_abs = self.buf_t0 + self.buf_secs
        self._arrival_perf = time.perf_counter()
        # Cleared here, set only by a real decode. Every early return below
        # used to leave the previous pass's cost in place, so the log printed
        # a constant stale number and the step planner fed on it.
        self.pass_secs = 0.0
        self.pass_buf_secs = 0.0
        self.pass_empty = False
        self.last_event = ""

        if self.buf_secs < self.min_secs:
            self.last_event = "min"
            return [], list(self.sp.unstable)

        # Idle skip. Only when nothing is pending and the buffer is silent, so
        # quiet-but-real speech is never discarded on amplitude alone -- a
        # fixed peak gate reads quiet compressed call audio as silence.
        if (self.silence_peak and not self.sp.unstable
                and float(np.abs(self.buf).max()) < self.silence_peak):
            self.last_event = "silence"
            self.buf_t0 += self.buf_secs
            self.buf = np.zeros(0, dtype=np.float32)
            return [], []

        # VAD gate: never hand Whisper audio with no speech in it. Music and
        # room tone are exactly what produce "Thank you for watching", and a
        # phantom lands at the head of the unstable tail where it can never
        # agree -- deadlocking output until the buffer cap. Silero separates
        # real narration (~85% of buffer) from music (0%) with a huge margin,
        # so skipping the model here is safe. Any pending tail is discarded
        # rather than committed, because it was decoded from that non-speech.
        #
        # Only judged on a buffer long enough to classify. A fresh 0.3s step
        # of ordinary speech often reports barely any speech, and gating on it
        # threw away real words (measured: 210 -> 194 words). During music the
        # buffer grows every step (nothing commits to cut it), so it clears
        # this bar within two steps anyway.
        if (self.buf_secs >= self.gate_min_buf_secs
                and self._speech_secs() < self.min_speech_secs):
            self.last_event = "gate"
            self.skipped_secs += self.buf_secs
            self.sp.discard()
            self.buf_t0 += self.buf_secs
            self.buf = np.zeros(0, dtype=np.float32)
            return [], []

        hyp = self._transcribe()

        if not hyp:
            # Whisper returned nothing: the audio was non-speech (the gate
            # above could not classify it yet) or every segment was dropped by
            # a confidence threshold. The pending tail can never be confirmed,
            # and the audio behind it is about to move on, so holding
            # unconfirmed words only guarantees a permanent prefix mismatch.
            self.last_event = "empty"
            self.sp.discard()
            # Keep the audio. Cutting it here would leave the buffer tiny, so
            # the VAD gate would never classify the run as non-speech and we
            # would feed Whisper a fresh sliver of music every step -- which
            # is what produces "Thank you for watching". Letting the buffer
            # grow is what lets the gate recognise music and stop calling the
            # model. Bound it so a genuinely-unusable buffer cannot grow
            # without limit.
            if self.buf_secs >= self.max_buf_secs:
                self.buf_t0 += self.buf_secs
                self.buf = np.zeros(0, dtype=np.float32)
            return [], []

        end_of_audio = self.buf_t0 + self.buf_secs
        paused = bool(hyp) and end_of_audio - hyp[-1].end >= self.pause_secs
        too_long = self.buf_secs >= self.max_buf_secs

        # Watchdog: a prefix match only commits while consecutive hypotheses
        # agree, so a head word that never settles blocks *all* output until
        # the buffer cap force-flushes (up to max_buf_secs of frozen captions).
        # Force the flush after stall_passes so a stall is bounded by a few
        # steps rather than by the cap.
        stuck = (self.sp.stalled >= self.stall_passes and self.sp.unstable)

        if paused or too_long or stuck:
            if stuck:
                self.last_event = "watchdog"
            elif too_long:
                self.last_event = "cap"
            else:
                self.last_event = "pause"
            committed = self.sp.flush(hyp)
            self._cut(end_of_audio if (paused or stuck or not committed)
                      else committed[-1].end)
            unstable = []
        else:
            before = self.sp.resyncs
            committed = self.sp.update(hyp)
            self.last_event = ("resync" if self.sp.resyncs > before
                               else ("commit" if committed else "none"))
            if committed:
                self._cut(committed[-1].end)
            unstable = list(self.sp.unstable)

        if committed and self.prompt_chars:
            self.history = (self.history + " " +
                            " ".join(w.text for w in committed)).strip()[-1000:]
        return committed, unstable

    def flush(self):
        """Commit whatever is still unstable (session end).

        `_cut` drops audio behind the last commit, so after a commit the
        buffer no longer holds the audio behind the *unstable* tail -- the
        words waiting on agreement. Re-decoding it is therefore the only way
        to recover them; without this they are lost on quit.
        """
        if self.buf_secs < 0.1:
            # Nothing to re-decode. The pending tail is nevertheless real and
            # was previously discarded here (`flush([])` commits nothing),
            # which is exactly how the last few spoken words vanished on quit.
            return self.sp.take_pending()
        return self.sp.flush(self._transcribe())
