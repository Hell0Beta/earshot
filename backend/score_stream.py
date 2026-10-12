"""Offline scoring for the rolling streamer.

Replays a WAV file through RollingStreamer at a fixed step size and scores the
committed text against the matching .srt. No loopback, no TUI, no GPU
contention from anything else -- this is the repeatable version of the
measurements in docs/streaming-latency.md.

Usage:
  python backend/score_stream.py AUDIO.wav REF.srt [--step 0.3] [--beam 5]
                            [--max-buf 12] [--pause 0.4] [--prompt-chars 0]

Reports WER, survival, dup5, time-to-first-word and pass-cost-vs-buffer.
"""

import argparse
import json
import math
import os
import re
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streaming import RollingStreamer  # noqa: E402

SR = 16000


# ---- reference / hypothesis normalisation ----

def norm_words(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return [w for w in s.split() if w]


def read_srt(path):
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    out = []
    for line in txt.splitlines():
        m = re.match(r"\d\d:\d\d:\d\d[,.]\d\d\d\s*-->\s*.*$", line.strip())
        if m:
            continue
        if "-->" in line:
            continue
        out.append(line)
    return norm_words(" ".join(out))


def read_wav(path):
    """Read a mono/stereo 16-bit WAV at any rate, returned as float32 16 kHz.

    The shipped narration is 24 kHz, so the harness resamples exactly the way
    the live path does (scipy resample_poly) rather than assuming 16 kHz.
    """
    import numpy as np
    from scipy.signal import resample_poly

    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())

    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    if rate != SR:
        g = math.gcd(rate, SR)
        x = resample_poly(x, SR // g, rate // g).astype(np.float32)
    return x


# ---- metrics ----

def levenshtein(ref, hyp):
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 100.0
    return 100.0 * levenshtein(ref, hyp) / len(ref)


def survival(ref, hyp):
    """Longest common subsequence ratio: how much of the reference survived."""
    if not ref:
        return 100.0
    prev = [0] * (len(hyp) + 1)
    for r in ref:
        cur = [0]
        for j, h in enumerate(hyp, 1):
            cur.append(prev[j - 1] + 1 if r == h else max(prev[j], cur[j - 1]))
        prev = cur
    return 100.0 * prev[-1] / len(ref)


def dup_n(words, n=5):
    seen, dups = set(), 0
    for i in range(len(words) - n + 1):
        g = tuple(words[i:i + n])
        if g in seen:
            dups += 1
        seen.add(g)
    return dups


# ---- run ----

def run(model, audio, step, beam, max_buf, pause, prompt_chars, dur=None):
    import numpy as np

    rs = RollingStreamer(model, max_buf_secs=max_buf, pause_secs=pause,
                         beam_size=beam, prompt_chars=prompt_chars,
                         min_secs=min(0.5, step))
    step_n = int(step * SR)
    committed, first_word, pass_costs, buf_costs = [], None, [], []
    t_wall = time.perf_counter()
    t_speech = 0.0
    for i in range(0, len(audio), step_n):
        chunk = audio[i:i + step_n].astype(np.float32)
        chunk_secs = len(chunk) / SR
        if dur and t_speech >= dur:
            break
        t_speech += chunk_secs
        c, _u = rs.feed(chunk)
        pass_costs.append(rs.pass_secs)
        buf_costs.append((rs.buf_secs, rs.pass_secs))
        if c and first_word is None:
            # seconds of audio consumed up to the moment the first word's
            # speech ended -- i.e. when it became visible in the live line
            first_word = t_speech - (chunk_secs - min(c[0].end, chunk_secs))
        committed.extend(c)
    tail = rs.flush()
    if tail and first_word is None:
        first_word = t_speech
    committed.extend(tail)
    wall = time.perf_counter() - t_wall

    # pass cost as a function of buffer length (for the cost table)
    buckets = {}
    for b, p in buf_costs:
        k = round(b, 1)
        buckets.setdefault(k, []).append(p)
    cost_by_buf = {k: sorted(v)[len(v) // 2] for k, v in buckets.items()
                   if len(v) >= 3}

    return {
        "text": " ".join(w.text for w in committed),
        "words": norm_words(" ".join(w.text for w in committed)),
        "n_words": len(committed),
        "audio_secs": t_speech,
        "wall_secs": wall,
        "rtf": wall / t_speech if t_speech else 0.0,
        "first_word_secs": first_word,
        "mean_pass": sum(pass_costs) / len(pass_costs) if pass_costs else 0.0,
        "max_pass": max(pass_costs) if pass_costs else 0.0,
        "pass_over_step": (max(pass_costs) / step) if pass_costs else 0.0,
        "cost_by_buf": cost_by_buf,
    }


def run_fixed(model, audio, window, beam, dur=None):
    """The shipped design, measured by this same scorer.

    Fixed non-overlapping windows, each decoded independently and appended --
    what earshot did before the rolling streamer. Included so the comparison
    is like-for-like: same WER/survival code, same reference, same machine.
    """
    import numpy as np

    win = int(window * SR)
    out, t_wall, t_speech = [], time.perf_counter(), 0.0
    costs = []
    for i in range(0, len(audio), win):
        chunk = audio[i:i + win].astype(np.float32)
        secs = len(chunk) / SR
        if dur and t_speech >= dur:
            break
        t_speech += secs
        t0 = time.perf_counter()
        segs, _ = model.transcribe(
            chunk, language="en", beam_size=beam, vad_filter=True,
            word_timestamps=True, condition_on_previous_text=False)
        costs.append(time.perf_counter() - t0)
        for s in segs:
            for w in (s.words or []):
                out.append(w.word.strip())
    text = " ".join(out)
    return {"text": text, "words": norm_words(text), "n_words": len(out),
            "audio_secs": t_speech, "wall_secs": time.perf_counter() - t_wall,
            "rtf": (time.perf_counter() - t_wall) / t_speech if t_speech else 0,
            "first_word_secs": None,
            "mean_pass": sum(costs) / len(costs) if costs else 0.0,
            "max_pass": max(costs) if costs else 0.0,
            "pass_over_step": (max(costs) / window) if costs else 0.0,
            "cost_by_buf": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("srt")
    ap.add_argument("--step", type=float, default=0.3)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-buf", type=float, default=12.0)
    ap.add_argument("--pause", type=float, default=0.4)
    ap.add_argument("--prompt-chars", type=int, default=0)
    ap.add_argument("--dur", type=float, default=0.0,
                    help="only score the first N seconds")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--mode", choices=["rolling", "fixed"], default="rolling",
                    help="rolling = LocalAgreement-2 streamer; fixed = the "
                         "shipped non-overlapping window baseline")
    ap.add_argument("--model-dir",
                    default=os.environ.get("EARSHOT_MODEL_DIR",
                                           r"D:\earshot\models"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cuda", compute_type="float16",
                         download_root=args.model_dir)

    audio = read_wav(args.wav)
    ref = read_srt(args.srt)

    if args.mode == "fixed":
        res = run_fixed(model, audio, args.step, args.beam, args.dur or None)
    else:
        res = run(model, audio, args.step, args.beam, args.max_buf,
                  args.pause, args.prompt_chars, args.dur or None)
    hyp = res.pop("words")
    res["wer"] = wer(ref, hyp)
    res["survival"] = survival(ref, hyp)
    res["dup5"] = dup_n(hyp, 5)
    res["ref_words"] = len(ref)
    res["hyp_words"] = len(hyp)
    res["step"] = args.step
    res["beam"] = args.beam

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("=" * 62)
    print("%s  step %.2fs  beam %d  %s"
          % (args.mode.upper(), args.step, args.beam,
             ("cap %.0fs  pause %.2fs  prompt %d"
              % (args.max_buf, args.pause, args.prompt_chars))
             if args.mode == "rolling" else "non-overlapping windows"))
    print("=" * 62)
    print("  audio          %.1fs   (%d ref words -> %d hyp words)"
          % (res["audio_secs"], res["ref_words"], res["hyp_words"]))
    print("  WER            %.1f%%" % res["wer"])
    print("  survival       %.1f%%" % res["survival"])
    print("  dup5           %d" % res["dup5"])
    print("  first word     %s"
          % ("%.2fs" % res["first_word_secs"]
             if res["first_word_secs"] else "n/a"))
    print("  wall           %.1fs  (%.2fx realtime)" % (res["wall_secs"], res["rtf"]))
    print("  pass cost      mean %.3fs  max %.3fs  (%.0f%% of step)"
          % (res["mean_pass"], res["max_pass"], 100 * res["pass_over_step"]))
    if res["cost_by_buf"]:
        print("  cost vs buffer:")
        for k in sorted(res["cost_by_buf"]):
            print("      %4.1fs buffer -> %.4fs pass" % (k, res["cost_by_buf"][k]))
    print()
    print("  TEXT: %s" % res["text"][:400])


if __name__ == "__main__":
    main()