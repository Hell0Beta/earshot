"""Unit tests for SentenceAssembler (no audio/GPU needed).

Run: ..\\.venv\\Scripts\\python.exe backend\\test_assembler.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from captions import SentenceAssembler


def test_first_block_cuts_at_200():
    a = SentenceAssembler()
    words = ("word%03d " % i for i in range(60))
    chunk = "".join(words).strip()  # ~480 chars
    finals = a.add_chunk(chunk, 0.0, 5.0)
    assert len(finals) == 2, finals  # 200 + chained remainder
    line1, t0, t1 = finals[0]
    assert len(line1) <= 200, len(line1)
    assert (t0, t1) == (0.0, 5.0)
    line2, _, _ = finals[1]
    assert line2.startswith(line1[-15:] + " "), line2[:30]
    assert len(line2) <= 200, len(line2)


def test_short_text_stays_buffered_then_flushes():
    a = SentenceAssembler()
    assert a.add_chunk("hello world", 1.0, 2.0) == []
    dtext, dt0 = a.draft()
    assert dtext == "hello world" and dt0 == 1.0, (dtext, dt0)
    out = a.flush()
    assert out is not None and out[0] == "hello world", out
    assert a.flush() is None  # second flush: nothing left


def test_chaining_math():
    a = SentenceAssembler(block_chars=200, chain_chars=15)
    a.add_chunk("x" * 200, 0.0, 1.0)  # hard cut, no spaces
    assert a.prev_tail == "x" * 15
    # chained target = 200 - 15 - 1 (joining space) = 184 fresh chars
    finals = a.add_chunk("y" * 184, 1.0, 2.0)
    assert len(finals) == 1, finals
    line = finals[0][0]
    assert line == "x" * 15 + " " + "y" * 184, line
    assert len(line) == 200, len(line)


def test_word_boundary_cut():
    a = SentenceAssembler(block_chars=40, chain_chars=5)
    finals = a.add_chunk("alpha beta gamma delta epsilon zeta eta theta iota kappa", 0.0, 1.0)
    assert finals, "expected a cut"
    assert not finals[0][0].endswith(" "), repr(finals[0][0])
    # remainder continues the sentence
    dtext, _ = a.draft()
    assert dtext, "remainder should be drafted"


def test_buffer_trims_chunk_overlap():
    a = SentenceAssembler()
    assert a.add_chunk("the quick brown fox jumps high", 0.0, 1.0) == []
    # next chunk restates the 4-word tail (case/punct variants still match):
    # only genuinely new words are appended
    assert a.add_chunk("brown, FOX jumps HIGH and dry", 1.0, 2.0) == []
    dtext, _ = a.draft()
    assert dtext == "the quick brown fox jumps high and dry", dtext
    # fewer than 3 shared words: no trim
    assert a.add_chunk("entirely unrelated phrase here", 3.0, 4.0) == []
    dtext, _ = a.draft()
    assert "unrelated phrase here" in dtext, dtext


def test_repeat_collapse_and_two_word_trim():
    a = SentenceAssembler()
    # adjacent duplicate tokens collapse
    assert a.add_chunk("this is is a test of from from speech", 0.0, 1.0) == []
    dtext, _ = a.draft()
    assert dtext == "this is a test of from speech", dtext
    # 2-word boundary repeat is now trimmed (was 3+)
    b = SentenceAssembler()
    assert b.add_chunk("we are capturing audio today", 0.0, 1.0) == []
    assert b.add_chunk("audio today and more", 1.0, 2.0) == []
    dtext, _ = b.draft()
    assert dtext == "we are capturing audio today and more", dtext
    # cross-boundary SINGLE repeat is dropped too
    c = SentenceAssembler()
    assert c.add_chunk("audio directly from the speakers", 0.0, 1.0) == []
    assert c.add_chunk("from the speakers without any cloud", 1.0, 2.0) == []
    dtext, _ = c.draft()
    assert dtext == "audio directly from the speakers without any cloud", dtext
    # ...but a repeated word later in the chunk is untouched
    d = SentenceAssembler()
    assert d.add_chunk("no", 0.0, 1.0) == []
    assert d.add_chunk("no I do not think so", 1.0, 2.0) == []
    dtext, _ = d.draft()
    assert dtext == "no I do not think so", dtext


def test_timestamps_advance_and_stay_ordered():
    a = SentenceAssembler(block_chars=60, chain_chars=10)
    words = " ".join("w%03d" % i for i in range(120))
    finals = []
    for k in range(6):
        # one second of speech per call, timestamps climbing
        part = " ".join(words.split()[k * 20:(k + 1) * 20])
        finals += a.add_chunk(part, float(k), float(k) + 1)
    assert len(finals) >= 4, len(finals)
    t0s = [f[1] for f in finals]
    t1s = [f[2] for f in finals]
    for i in range(1, len(t0s)):
        assert t0s[i] >= t0s[i - 1], "t0 went backwards: %s" % t0s
        assert t1s[i] >= t1s[i - 1], "t1 went backwards: %s" % t1s
    # the stale-timestamp bug froze every block at the first value
    assert len(set(t0s)) > 1, "all blocks share one timestamp: %s" % t0s
    assert t0s[-1] > t0s[0], "final block should start later than the first"
    # and the last block must not claim to start before the session began
    assert all(t >= 0 for t in t0s + t1s), t0s + t1s


def test_empty_chunks_ignored():
    a = SentenceAssembler()
    assert a.add_chunk("   ", 0.0, 1.0) == []
    assert a.draft() == ("", 0.0)
    assert a.flush() is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok:", t.__name__)
    print("ALL %d PASSED" % len(tests))
