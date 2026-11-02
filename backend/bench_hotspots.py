"""Benchmark decode cost on the Zoom hot-spot fixtures.

bench_pass.py measured the cost cliff on clean narration. This variant runs
the same lazy-generator-correct measurement on the six reproducible slow-pass
spans sliced from the Oct 23 Zoom session (backend/testdata/oct23zoom/
hotslices.wav), where the live session and four replays all saw passes of
1.5-12.9s on <2s buffers, and repeats each config N times to capture the
run-to-run spread (p10/p90/max) that distinguishes cuDNN nondeterminism from
a stable cost.

Usage:
  ..\\.venv\\Scripts\\python.exe backend\\bench_hotspots.py
      [--audio PATH] [--runs 8] [--model small.en]
      [--beam 1] ... [--beam 5] ... [--compute float16 int8_float16 int8]
      [--model-dir D:\\earshot\\models]
"""

import argparse
import statistics
import sys
import time

sys.path.insert(0, __import__("os").path.dirname(__file__))

from score_stream import read_wav  # noqa: E402


def transcribe_once(model, audio, beam, word_ts=True):
    """Consume the generator - timing the call alone reports ~3ms."""
    segs, _ = model.transcribe(
        audio, language="en", beam_size=beam, vad_filter=True,
        word_timestamps=word_ts)
    n = sum(1 for s in segs for _ in (s.words or []))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None,
                    help="default: testdata/oct23zoom/hotslices.wav")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--beam", type=int, action="append", default=[])
    ap.add_argument("--compute", type=str, action="append", default=[])
    ap.add_argument("--model-dir", default=r"D:\earshot\models")
    args = ap.parse_args()

    import os

    beams = args.beam or [1]
    computes = args.compute or ["float16"]

    from faster_whisper import WhisperModel

    default_audio = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "testdata",
        "oct23zoom", "hotslices.wav")
    audio_f32 = read_wav(args.audio or default_audio)

    for compute in computes:
        model = WhisperModel(args.model, device="cuda", compute_type=compute,
                             download_root=args.model_dir)
        for beam in beams:
            times, words = [], []
            for i in range(args.runs):
                t0 = time.perf_counter()
                n = transcribe_once(model, audio_f32, beam)
                times.append(time.perf_counter() - t0)
                words.append(n)
            times.sort()
            q = lambda p: times[min(len(times) - 1, int(p * len(times)))]  # noqa: E731
            print("beam %d  %-13s  n=%d  median %.3fs  p10 %.3fs  p90 %.3fs"
                  "  max %.3fs  words %d"
                  % (beam, compute, args.runs, statistics.median(times),
                     q(0.1), q(0.9), times[-1], max(words)))
        del model


if __name__ == "__main__":
    main()
