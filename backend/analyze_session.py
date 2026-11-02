"""Reference-free analysis of one earshot session capture.

Reads the four artifacts the Go frontend writes beside a transcript:
  <stem>.md           live caption text ([" - [00:16] ..."] lines)
  <stem>.txt          LLM-polished voice-to-text document
  <stem>.latency.csv  per-word + per-event telemetry rows
  <stem>.backend.log  per-step loop lines + SLOWPASS marks

Reports accuracy proxies (hallucination artifacts, dup5, words-per-minute)
and latency diagnostics (pass-cost distribution vs buffer, slow-pass
attribution, word-latency drift over the session). No reference transcript
is required, so nothing here is a WER claim.

Usage:
  python backend/analyze_session.py backend/testdata/oct23zoom/2026-10-23_0915
        [--slow-pass 1.5]
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_stream import norm_words, dup_n  # noqa: E402

MD_LINE = re.compile(r"^- \[(\d\d):(\d\d)\] (.*)$")
STEP_LINE = re.compile(
    r"\[([\d:]+) step (\d+)\] (\S+)\s+target ([\d.]+)s pass ([\d.]+)s "
    r"buffer ([\d.]+)s backlog ([\d.]+)s -> \+(\d+) committed, (\d+) unstable")
SLOWPASS_LINE = re.compile(
    r"\[([\d:]+) step (\d+)\] SLOWPASS ([\d.]+)s on ([\d.]+)s of audio"
    r"(, no words)? \(event=(\w+)\)")

ARTIFACTS = [
    "thank you for watching", "thanks for watching",
    "thanks a lot", "thank you very much",
    "amara org", "transcription by", "transcribed by",
    "subtitles by", "like and subscribe", "please subscribe",
]


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[i]


def stats(vals):
    s = sorted(vals)
    return {
        "n": len(s),
        "median": percentile(s, 0.50),
        "p90": percentile(s, 0.90),
        "max": s[-1] if s else 0.0,
        "sum": sum(s),
    }


# ---- latency.csv ----

def parse_csv(path):
    """Returns (word_rows, event_rows_by_name)."""
    words, events = [], defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {
                "wall": float(r["wall"]),
                "cycle": int(r["cycle"]),
                "event": r["event"],
                "word": r["word"],
                "audio_end_s": float(r["audio_end_s"]),
                "latency_s": float(r["latency_s"]),
                "pass_s": float(r["pass_s"]),
                "target_s": float(r["target_s"]),
                "buffer_s": float(r["buffer_s"]),
            }
            if r["event"] == "word":
                words.append(row)
            else:
                events[r["event"]].append(row)
    return words, events


def attr_by_event(rows):
    """Group non-word rows that actually decoded (pass_s > 0) by event."""
    out = defaultdict(lambda: {"n": 0, "last_pass": 0.0})
    for r in rows:
        if r["pass_s"] <= 0:
            continue
        e = out[r["event"]]
        e["n"] += 1
        e["last_pass"] = r["pass_s"]  # CPU cost of a gate skip is 0, so the
        # last decode seen is the best estimate of the pass this event paid.
    return out


# ---- backend.log ----

def parse_log(path):
    steps, slow = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_LINE.match(line.strip())
            if m:
                steps.append({
                    "clock": m.group(1), "cycle": int(m.group(2)),
                    "event": m.group(3), "target": float(m.group(4)),
                    "pass": float(m.group(5)), "buffer": float(m.group(6)),
                    "backlog": float(m.group(7)),
                    "committed": int(m.group(8)), "unstable": int(m.group(9)),
                })
                continue
            m = SLOWPASS_LINE.match(line.strip())
            if m:
                slow.append({
                    "clock": m.group(1), "cycle": int(m.group(2)),
                    "pass": float(m.group(3)), "buffer": float(m.group(4)),
                    "nowords": bool(m.group(5)), "event": m.group(6),
                })
    return steps, slow


# ---- .md text ----

def parse_md(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = MD_LINE.match(line.strip())
            if m:
                lines.append((int(m.group(1)) * 60 + int(m.group(2)),
                              m.group(3)))
    return lines


def text_metrics(md_lines):
    all_words, minute_words = [], defaultdict(int)
    for stamp, text in md_lines:
        w = norm_words(text)
        all_words.extend(w)
        minute_words[stamp // 60] += len(w)
    low = " ".join(all_words)
    artifacts = [(a, low.count(a)) for a in ARTIFACTS if low.count(a) > 0]
    artifacts.sort(key=lambda t: -t[1])
    # word-rate sanity: lecture speech is ~110-180 wpm; gauge drift per minute
    hist = {m: minute_words[m] for m in sorted(minute_words)}
    return {
        "n_words": len(all_words),
        "dup5": dup_n(all_words, 5),
        "artifacts": artifacts,
        "artifact_total": sum(n for _, n in artifacts),
        "wpm": hist,
    }


def md_txt_drift(md_lines, txt_path):
    """How much the polish pass changed the text: LCS survival both ways."""
    with open(txt_path, encoding="utf-8") as f:
        txt = f.read()
    a = norm_words(" ".join(t for _, t in md_lines))
    b = norm_words(txt)
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    both = sum(bl.size for bl in sm.get_matching_blocks())
    return {
        "md_words": len(a), "txt_words": len(b),
        "shared": both,
        "md_survival": 100.0 * both / len(a) if a else 0.0,
        "txt_survival": 100.0 * both / len(b) if b else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="path prefix without extension")
    ap.add_argument("--slow-pass", type=float, default=1.5)
    args = ap.parse_args()
    d = os.path.dirname(args.stem) or "."
    stem = os.path.basename(args.stem)

    words, events = parse_csv(os.path.join(d, stem + ".latency.csv"))
    steps, slow = parse_log(os.path.join(d, stem + ".backend.log"))
    md_lines = parse_md(os.path.join(d, stem + ".md"))
    tm = text_metrics(md_lines)
    txt_path = os.path.join(d, stem + ".txt")
    drift = (md_txt_drift(md_lines, txt_path)
             if os.path.exists(txt_path) else None)

    t0 = min((r["wall"] for r in sum(events.values(), []) + words),
             default=0.0)

    print("=" * 62)
    print("session %s" % stem)
    print("=" * 62)

    print("\n-- live text (.md) --")
    print("  caption words       %d" % tm["n_words"])
    if md_lines:
        span = max(s for s, _ in md_lines)
        print("  caption minute span %.0f min %02ds" % (span // 60, span % 60))
    print("  dup5                %d" % tm["dup5"])
    print("  halluc. artifacts   %d total" % tm["artifact_total"])
    for a, n in tm["artifacts"]:
        print("      %-28s x%d" % (a, n))

    if drift:
        print("\n-- polish pass (.md vs .txt) --")
        print("  words  %d -> %d   shared %d" %
              (drift["md_words"], drift["txt_words"], drift["shared"]))
        print("  survival of .md in .txt   %.1f%%" % drift["md_survival"])
        print("  survival of .txt in .md   %.1f%% (invention > 100-m"
              "survival means the LLM added text)" % drift["txt_survival"])

    # word-latency over the session, in 10-minute buckets
    buckets = defaultdict(list)
    for r in words:
        buckets[int((r["wall"] - t0) / 600)].append(r["latency_s"])
    print("\n-- word latency drift (10-min buckets) --")
    for k in sorted(buckets):
        s = stats(buckets[k])
        print("  %2d-%2dm  n=%4d  median %.2fs  p90 %.2fs  max %.2fs"
              % (k * 10, k * 10 + 10, s["n"], s["median"], s["p90"],
                 s["max"]))
    s = stats([r["latency_s"] for r in words])
    print("  all      n=%4d  median %.2fs  p90 %.2fs  max %.2fs"
          % (s["n"], s["median"], s["p90"], s["max"]))

    decodes = {}
    for r in sum(events.values(), []) + words:
        if r["pass_s"] > 0:
            decodes[r["cycle"]] = r  # several rows share one cycle
    decodes = list(decodes.values())
    s = stats([r["pass_s"] for r in decodes])
    print("\n-- decode pass cost (rows with pass_s>0) --")
    print("  n=%d  median %.3fs  p90 %.3fs  max %.3fs  total %.1fs"
          % (s["n"], s["median"], s["p90"], s["max"], s["sum"]))

    # cost vs buffer
    by_buf = defaultdict(list)
    for r in decodes:
        by_buf[round(r["buffer_s"])].append(r["pass_s"])
    print("\n  cost vs buffer (pass_s>0):")
    for k in sorted(by_buf):
        v = sorted(by_buf[k])
        if len(v) < 3:
            continue
        print("      %4.1fs buffer -> median %.3fs  p90 %.3fs  (%d rows)"
              % (k, percentile(v, 0.5), percentile(v, 0.9), len(v)))

    # slow decode attribution
    sp = [r for r in decodes if r["pass_s"] >= args.slow_pass]
    print("\n-- slow passes >= %.1fs: %d, %.1fs total --"
          % (args.slow_pass, len(sp), sum(r["pass_s"] for r in sp)))
    print("  logged SLOWPASS lines: %d  (%.1fs total, %d no-words)"
          % (len(slow), sum(x["pass"] for x in slow),
             sum(1 for x in slow if x["nowords"])))
    for r in sorted(sp, key=lambda r: -r["pass_s"])[:12]:
        print("  +%.1fs  cycle %d  event %-7s buffer %.1fs  pass %.2fs"
              % (r["wall"] - t0, r["cycle"], r["event"], r["buffer_s"],
                 r["pass_s"]))

    # event accounting (log is authoritative for even zero-pass events)
    ev = defaultdict(int)
    for st in steps:
        ev[st["event"]] += 1
    print("\n-- loop events (from %d step lines) --" % len(steps))
    for k in sorted(ev, key=lambda k: -ev[k]):
        print("  %-9s %d" % (k, ev[k]))
    comm = sum(st["committed"] for st in steps)
    stall = sum(st["pass"] for st in steps)
    print("  committed %d words over %.1fs of pass time" % (comm, stall))


if __name__ == "__main__":
    main()
