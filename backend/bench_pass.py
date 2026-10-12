"""Measure the TRUE cost of one transcription pass.

Two traps this exists to avoid, both of which produced wrong numbers before:

1. `model.transcribe()` returns a **lazy generator**. Timing the call without
   consuming it measures almost nothing -- the model never runs. Every number
   here consumes the generator.
2. faster-whisper pads to `chunk_length` (30s) internally, so a pass over a
   0.3s buffer can cost the same as one over 12s. Cost is therefore roughly
   buffer-independent, and the step budget must be set from the *measured*
   pass time, not the buffer length.

Usage:
  python backend/bench_pass.py [--audio WAV] [--repeats 5]
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from score_stream import read_wav  # noqa: E402

SR = 16000
DEFAULT_AUDIO = (r"D:\narrated-animations\videos\001_genesis-4"
                 r"\voice\narration.wav")


def timed(fn, repeats):
    fn()  # warm up (CUDA autotune, lazy init) -- excluded from the numbers
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=DEFAULT_AUDIO)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--model-dir", default=os.environ.get(
        "EARSHOT_MODEL_DIR", r"D:\earshot\models"))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sizes", default="0.3,0.6,1,2,4,8,12")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    model = WhisperModel(args.model, device="cuda", compute_type="float16",
                         download_root=args.model_dir)
    audio = read_wav(args.audio)
    vad = VadOptions(min_silence_duration_ms=150, speech_pad_ms=50)

    def call(chunk, **kw):
        segs, _ = model.transcribe(
            chunk, language="en", vad_filter=True, word_timestamps=True,
            condition_on_previous_text=False,
            vad_parameters=dict(min_silence_duration_ms=150,
                                speech_pad_ms=50),
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0, **kw)
        return sum(1 for _ in segs)      # MUST consume: it is lazy

    sizes = [float(s) for s in args.sizes.split(",")]
    print("=" * 74)
    print("TRUE PER-PASS COST (generator consumed).  step budget at 0.3s = 300ms")
    print("=" * 74)
    print("buffer_s   beam1_ms   beam5_ms   p90_beam5   vs 300ms budget")
    rows = []
    for s in sizes:
        n = int(s * SR)
        chunk = audio[:n].astype(np.float32)
        b1 = timed(lambda: call(chunk, beam_size=1), args.repeats)
        b5 = timed(lambda: call(chunk, beam_size=5), args.repeats)
        p90 = statistics.quantiles(b5, n=10)[-1] if len(b5) > 1 else b5[0]
        med1, med5 = statistics.median(b1), statistics.median(b5)
        rows.append((s, med1, med5, p90))
        print("%7.1f  %9.1f  %9.1f  %10.1f   %6.0f%%"
              % (s, med1, med5, p90, 100 * p90 / 300.0))

    v = timed(lambda: get_speech_timestamps(audio[:SR].astype(np.float32), vad),
              args.repeats)
    print()
    print("VAD alone (1s buffer)   median %.1f ms" % statistics.median(v))
    worst = max(r[3] for r in rows)
    print("WORST p90 pass          %.1f ms  -> minimum viable step %.2fs"
          % (worst, worst / 1000.0 * 1.25))
    print()
    print("NOTE: cost is near-flat across buffer sizes because faster-whisper")
    print("pads to chunk_length=30s internally. The step must be sized from")
    print("this measurement, not from --step-secs.")


if __name__ == "__main__":
    main()