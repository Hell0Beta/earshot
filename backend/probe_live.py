"""Measure the two streaming defects on real audio: stall + hallucination.

No reference transcript needed. For each pass we record what the streamer
did, which lets us attribute a gap in the committed text to one of two very
different causes:

  FLUSH   the tail was force-committed (pause or buffer cap) -- output moved
  DEADLOCK nothing committed and the buffer was NOT cut -- the prefix rule
          is stuck on a head disagreement

The second is a bug: it freezes the captions until the buffer cap (12s) fires.
The first is the design working as intended.

Usage:
  python backend/probe_live.py AUDIO.wav [--step 0.3] [--pause 0.2]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from score_stream import norm_words, read_wav  # noqa: E402
from streaming import RollingStreamer  # noqa: E402

SR = 16000

# Whisper's stock end-of-speech / non-speech artifacts. Only multi-word stock
# phrases and unmistakably non-speech sign-offs count -- single common words
# like "you" or "okay" are legitimate transcript vocabulary and scoring them
# as artifacts just measures how much speech got transcribed.
ARTIFACTS = [
    "thank you for watching", "thank you very much", "thanks for watching",
    "thanks a lot", "thank you", "bye", "amara org", "transcription by",
    "transcribed by", "subtitles by", "subscribe", "like and subscribe",
    "please subscribe",
]


class Probe(RollingStreamer):
    """Records per-pass state so stalls can be attributed after the run."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.passes = []      # dict per feed()
        self.commits = []     # (audio_time, text)

    def feed(self, audio):
        t_before = self.buf_t0
        buf_before = self.buf_secs
        streak_before = len(self.sp.unstable)
        committed, unstable = super().feed(audio)
        t = self._t + len(audio) / SR
        self._t = t
        cut = self.buf_t0 - t_before
        self.passes.append({
            "t": t,
            "n_committed": len(committed),
            "n_unstable": len(unstable),
            "buf_before": buf_before,
            "buf_after": self.buf_secs,
            "cut": cut,
            "pass_secs": self.pass_secs,
            "streak_before": streak_before,
        })
        if committed:
            self.commits.append((t, " ".join(w.text for w in committed)))
        return committed, unstable

    _t = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--step", type=float, default=0.3)
    ap.add_argument("--pause", type=float, default=0.2)
    ap.add_argument("--max-buf", type=float, default=12.0)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="word probability floor (0 = off)")
    ap.add_argument("--stall", type=float, default=1.0,
                    help="gap in committed text that counts as a stall")
    ap.add_argument("--vad-min-silence-ms", type=int, default=150)
    ap.add_argument("--vad-pad-ms", type=int, default=50)
    ap.add_argument("--min-speech-secs", type=float, default=0.10,
                    help="VAD gate: skip the model below this much speech. "
                         "0 disables the gate (pre-fix behaviour)")
    ap.add_argument("--stall-passes", type=int, default=6,
                    help="passes with no commit before force-flushing. "
                         "9999 effectively disables the watchdog")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--model-dir", default=os.environ.get(
        "EARSHOT_MODEL_DIR", r"D:\earshot\models"))
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cuda", compute_type="float16",
                         download_root=args.model_dir)

    audio = read_wav(args.wav)
    rs = Probe(model, max_buf_secs=args.max_buf, pause_secs=args.pause,
               beam_size=args.beam, prompt_chars=0,
               min_secs=min(0.5, args.step), min_prob=args.min_prob,
               min_speech_secs=args.min_speech_secs,
               stall_passes=args.stall_passes)

    step_n = int(args.step * SR)
    t0 = time.perf_counter()
    for i in range(0, len(audio), step_n):
        rs.feed(audio[i:i + step_n].astype(np.float32))
    tail = rs.flush()
    if tail:
        rs.commits.append((rs._t, " ".join(w.text for w in tail)))
    wall = time.perf_counter() - t0

    full_text = " ".join(c for _, c in rs.commits)
    words = norm_words(full_text)

    # ---- commit gaps: time between successive committed passes ----
    gaps = []
    last_commit_t = 0.0
    for t, _txt in rs.commits:
        gaps.append((t - last_commit_t, t, last_commit_t))
        last_commit_t = t
    gap_secs = sorted(g[0] for g in gaps)

    # ---- speech regions: a gap spanning silence is NOT a stall ----
    # This file is ~140s of music between narration. Nothing can be committed
    # during music, so long gaps there are correct behaviour. Only gaps that
    # fall inside speech count as stalls.
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    spans = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=args.vad_min_silence_ms,
                          speech_pad_ms=args.vad_pad_ms))
    regions = [(d["start"] / SR, d["end"] / SR) for d in spans]

    def in_speech(t, slack=0.0):
        return any(a - slack <= t <= b + slack for a, b in regions)

    speech_gaps = []      # (gap, t, prev_t) tuples inside speech
    silence_gaps = []     # (gap, t, prev_t) tuples spanning music
    for gap, t, prev in gaps:
        # The gap runs from `prev` to `t`. If either end is in speech, the
        # silence was not the reason we were quiet.
        (speech_gaps if (in_speech(prev) or in_speech(t))
         else silence_gaps).append((gap, t, prev))

    speech_gap_secs = sorted(g for g, _, _ in speech_gaps)
    stalls = [x for x in speech_gaps if x[0] >= args.stall]

    def pass_at(t):
        best = None
        for p in rs.passes:
            if p["t"] <= t + 1e-6:
                best = p
            else:
                break
        return best

    deadlock, flushed = [], []
    for gap, t, _prev in stalls:
        p = pass_at(t)
        # A cut larger than the step means the tail was flushed past the
        # ordinary trim; no cut across the gap means nothing advanced.
        if p is not None and p["cut"] > args.step * 1.5:
            flushed.append(gap)
        else:
            deadlock.append(gap)

    # ---- artifact scan on committed text ----
    joined = " " + " ".join(words) + " "
    hits = {}
    for a in ARTIFACTS:
        c = joined.count(" %s " % a)
        if c:
            hits[a] = c

    # ---- buffer / cost profile ----
    bufs = [p["buf_after"] for p in rs.passes]
    costs = [p["pass_secs"] for p in rs.passes]

    res = {
        "audio_secs": rs._t,
        "wall_secs": wall,
        "rtf": wall / rs._t if rs._t else 0,
        "passes": len(rs.passes),
        "commits": len(rs.commits),
        "words": len(words),
        "gap_median": gap_secs[len(gap_secs) // 2] if gap_secs else 0,
        "gap_p90": gap_secs[int(len(gap_secs) * 0.9)] if gap_secs else 0,
        "gap_max": gap_secs[-1] if gap_secs else 0,
        "speech_gap_median": (speech_gap_secs[len(speech_gap_secs) // 2]
                              if speech_gap_secs else 0),
        "speech_gap_max": speech_gap_secs[-1] if speech_gap_secs else 0,
        "silence_gaps": len(silence_gaps),
        "skipped_secs": rs.skipped_secs,
        "stalls": len(stalls),
        "stall_deadlock": len(deadlock),
        "stall_flush": len(flushed),
        "deadlock_max": max(deadlock) if deadlock else 0,
        "deadlock_total": sum(deadlock),
        "artifacts": hits,
        "artifact_total": sum(hits.values()),
        "buf_max": max(bufs) if bufs else 0,
        "buf_mean": sum(bufs) / len(bufs) if bufs else 0,
        "pass_max": max(costs) if costs else 0,
        "pass_mean": sum(costs) / len(costs) if costs else 0,
        "steps": {"step": args.step, "pause": args.pause,
                  "max_buf": args.max_buf, "min_prob": args.min_prob,
                  "beam": args.beam},
    }

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("=" * 66)
    print("LIVE PROBE  %s" % os.path.basename(args.wav))
    print("step %.2f  pause %.2f  cap %.0f  beam %d  min_prob %.2f"
          % (args.step, args.pause, args.max_buf, args.beam, args.min_prob))
    print("=" * 66)
    print("  audio %.1fs   passes %d   commits %d   words %d"
          % (res["audio_secs"], res["passes"], res["commits"], res["words"]))
    print("  wall  %.1fs  (%.2fx realtime)" % (res["wall_secs"], res["rtf"]))
    print()
    print("  COMMIT GAPS (all)   median %.2fs  p90 %.2fs  max %.2fs"
          % (res["gap_median"], res["gap_p90"], res["gap_max"]))
    print("  SPEECH-ONLY GAPS    median %.2fs  max %.2fs   (%d gaps)"
          % (res["speech_gap_median"], res["speech_gap_max"],
             len(speech_gaps)))
    print("  non-speech gaps     %d  (expected: the audio is music there)"
          % res["silence_gaps"])
    print("  VAD skipped         %.1fs of audio as non-speech"
          % res["skipped_secs"])
    print()
    print("  STALLS DURING SPEECH (gap >= %.1fs): %d" % (args.stall, res["stalls"]))
    print("     deadlock (BUG)      %d   total %.1fs   max %.2fs"
          % (res["stall_deadlock"], res["deadlock_total"],
             res["deadlock_max"]))
    print("     flush (by design)   %d" % res["stall_flush"])
    print()
    print("  ARTIFACTS in committed text: %d" % res["artifact_total"])
    for a, c in sorted(hits.items(), key=lambda kv: -kv[1]):
        print("     %-24s x%d" % (a, c))
    print()
    print("  BUFFER  mean %.2fs  max %.2fs (cap %.0fs)"
          % (res["buf_mean"], res["buf_max"], args.max_buf))
    print("  PASS    mean %.4fs  max %.4fs"
          % (res["pass_mean"], res["pass_max"]))


if __name__ == "__main__":
    main()
