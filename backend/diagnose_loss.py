"""Attribute missing reference words to either the streamer or the model.

For every word in the reference that never reaches the committed output,
decide which of these it was:

  LOST_IN_STREAMER  the model DID decode it (it appeared in some hypothesis,
                    committed or unstable) but it was never committed. Our bug.
  NEVER_DECODED     the model never produced it in any pass. Model limitation --
                    no windowing scheme can recover it.

Only the first kind is fixable here. Run:

  python backend/diagnose_loss.py AUDIO.wav REF.srt [--step 0.3] [--pause 0.2]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from score_stream import SR, norm_words, read_srt, read_wav  # noqa: E402
from streaming import RollingStreamer, Word  # noqa: E402


class TracingStreamer(RollingStreamer):
    """Records every word the model produces, and what actually commits.

    Commits happen inside StablePrefix, which `feed`/`flush` call directly, so
    the hook has to live on the StablePrefix -- overriding `_commit` here would
    never fire and would report zero committed words.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []          # every word from every hypothesis
        self.committed = []     # words that actually reached the output
        self.discarded = []     # words dropped by the boundary filter
        sp = self.sp

        orig_commit = sp._commit
        orig_fresh = sp._fresh

        def commit(words):
            self.committed.extend(words)
            return orig_commit(words)

        def fresh(hyp):
            out = orig_fresh(hyp)
            self.discarded.extend(w for w in hyp if w not in out)
            return out

        sp._commit = commit
        sp._fresh = fresh

    def _transcribe(self):
        hyp = super()._transcribe()
        self.seen.extend(hyp)
        return hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("srt")
    ap.add_argument("--step", type=float, default=0.3)
    ap.add_argument("--pause", type=float, default=0.2)
    ap.add_argument("--max-buf", type=float, default=12.0)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--dur", type=float, default=0.0)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--model-dir", default=os.environ.get(
        "EARSHOT_MODEL_DIR", r"D:\earshot\models"))
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cuda", compute_type="float16",
                         download_root=args.model_dir)

    audio = read_wav(args.wav)
    ref = read_srt(args.srt)

    rs = TracingStreamer(model, max_buf_secs=args.max_buf,
                         pause_secs=args.pause, beam_size=args.beam,
                         prompt_chars=0, min_secs=min(0.5, args.step))
    step_n = int(args.step * SR)
    t = 0.0
    for i in range(0, len(audio), step_n):
        chunk = audio[i:i + step_n].astype(np.float32)
        t += len(chunk) / SR
        if args.dur and t >= args.dur:
            break
        rs.feed(chunk)
    rs.flush()

    committed = [w.text for w in rs.committed]
    committed_n = norm_words(" ".join(committed))
    seen_n = norm_words(" ".join(w.text for w in rs.seen))

    ref_set, seen_set = set(ref), set(seen_n)
    missing = [w for w in ref if w not in set(committed_n)]

    lost = [w for w in missing if w in seen_set]
    never = [w for w in missing if w not in seen_set]

    print("=" * 64)
    print("LOSS ATTRIBUTION  (%d reference words)" % len(ref))
    print("=" * 64)
    print("  committed words          %d" % len(committed_n))
    print("  reference words          %d" % len(ref))
    print("  reference words missing  %d  (%.1f%%)"
          % (len(missing), 100.0 * len(missing) / len(ref)))
    print()
    print("  LOST_IN_STREAMER  %d  (%.1f%% of ref)  <- OUR BUG, fixable"
          % (len(lost), 100.0 * len(lost) / len(ref)))
    print("  NEVER_DECODED     %d  (%.1f%% of ref)  <- model limitation"
          % (len(never), 100.0 * len(never) / len(ref)))
    print()
    print("  words decoded at least once   %d unique of %d"
          % (len(seen_set), len(ref_set)))
    print("  reference words NEVER decoded %d  (%.1f%%)"
          % (len(ref_set - seen_set), 100.0 * len(ref_set - seen_set) / len(ref)))
    print()

    if lost:
        from collections import Counter
        print("  LOST_IN_STREAMER examples (decoded but never committed):")
        for w, n in Counter(lost).most_common(args.show):
            print("      %-18s x%d" % (w, n))
    if never:
        from collections import Counter
        print()
        print("  NEVER_DECODED examples (model never produced these):")
        for w, n in Counter(never).most_common(args.show):
            print("      %-18s x%d" % (w, n))

    print()
    print("  discard events (dropped by _fresh boundary filter): %d"
          % len(rs.discarded))
    print("  hypotheses run: %d   total word-decodes: %d"
          % (len(rs.seen), len(rs.seen)))


if __name__ == "__main__":
    main()
