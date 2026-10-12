"""Unit tests for SentenceAssembler and the live-line composer.

Run: ..\\.venv\\Scripts\\python.exe backend\\test_assembler.py

The assembler receives only *committed* words from the streamer, so there is
no overlap-trimming behaviour here: `add_chunk` appends fresh text and cuts
blocks when the buffer fills. The contracts that matter are that no input word
is ever lost, and that the live (`draft`) line shows the whole pending block
rather than just the newest word -- both are regression guards for bugs that
shipped visibly broken once already.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from captions import SentenceAssembler, compose_live_line


class W:
    """Minimal stand-in for streaming.Word (only `.text` and `.start`)."""

    def __init__(self, text, start=0.0):
        self.text = text
        self.start = start


def words_of(text):
    return text.split()


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


def test_block_chars_200_chain_math():
    """Default block is 200. Every line stays within it and chains 15 runes."""
    a = SentenceAssembler(block_chars=200, chain_chars=15)
    words = " ".join("w%03d" % i for i in range(120))
    finals = []
    for k in range(8):
        part = " ".join(words.split()[k * 15:(k + 1) * 15])
        finals += a.add_chunk(part, float(k), float(k) + 1)
    assert finals, "expected at least one commit"
    for line, _, _ in finals:
        assert len(line) <= 200, len(line)
    assert len(finals[0][0]) >= 195, len(finals[0][0])
    for i in range(1, len(finals)):
        prev = finals[i - 1][0][-15:].strip()
        assert finals[i][0].startswith(prev), (finals[i][0][:40], prev)


def test_word_boundary_cut():
    a = SentenceAssembler(block_chars=40, chain_chars=5)
    finals = a.add_chunk("alpha beta gamma delta epsilon zeta eta theta iota kappa", 0.0, 1.0)
    assert finals, "expected a cut"
    assert not finals[0][0].endswith(" "), repr(finals[0][0])
    dtext, _ = a.draft()
    assert dtext, "remainder should be drafted"


def test_timestamps_advance_and_stay_ordered():
    a = SentenceAssembler(block_chars=60, chain_chars=10)
    words = " ".join("w%03d" % i for i in range(120))
    finals = []
    for k in range(6):
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
    assert all(t >= 0 for t in t0s + t1s), t0s


def test_empty_chunks_ignored():
    a = SentenceAssembler()
    assert a.add_chunk("   ", 0.0, 1.0) == []
    assert a.draft() == ("", 0.0)
    assert a.flush() is None


# ---- content-loss regression ----
#
# `_collapse_intra` once deleted everything between two copies of a repeated
# 4-gram, which on recited text destroyed real speech ("and he said unto her
# and he said unto her" -> "and he said unto her"). Word *counts* looked fine
# and duplication counts went down, so nothing caught it until a ground-truth
# reference existed. These tests assert the words survive.


def test_repeated_sentences_are_not_collapsed():
    a = SentenceAssembler()
    a.add_chunk("The LORD God said and the LORD God said unto the woman", 0.0, 3.0)
    dtext, _ = a.draft()
    assert dtext.count("LORD God said") == 2, dtext
    assert "unto the woman" in dtext, dtext


def test_repeated_clauses_survive_a_commit():
    """A repeated phrase spanning a commit must not lose words."""
    a = SentenceAssembler(block_chars=80, chain_chars=10)
    chunk = ("then the lord said to cain and then the lord said to cain "
             "and then the lord said to cain")
    finals = a.add_chunk(chunk, 0.0, 3.0)
    tail = a.flush()
    if tail:
        finals.append(tail)
    out = " ".join(l for l, _, _ in finals)
    # All three repetitions must survive the block cuts.
    assert out.count("then the lord said") == 3, out
    assert out.count("to cain") == 3, out


def test_no_words_are_lost_across_a_long_feed():
    """Every distinct input word must appear in the output or the buffer."""
    a = SentenceAssembler(block_chars=120, chain_chars=15)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    for k in range(20):
        a.add_chunk(" ".join(vocab), float(k), float(k) + 1)
    tail = a.flush()
    finals = []
    a2 = SentenceAssembler(block_chars=120, chain_chars=15)
    for k in range(20):
        finals += a2.add_chunk(" ".join(vocab), float(k), float(k) + 1)
    t = a2.flush()
    if t:
        finals.append(t)
    out = " ".join(l for l, _, _ in finals)
    for w in vocab:
        assert w in out, "%r was lost: %r" % (w, out)
    # 140 words in; the output must be within a sane multiple (chaining adds
    # a 15-rune prefix per line, duplication would inflate far more).
    assert len(out.split()) <= 140 * 1.6, len(out.split())


# ---- live line (draft) composition ----
#
# Regression guard: an earlier version built the live line from the unstable
# tail alone, so the display showed one word at a time and the sentence only
# appeared once the block filled and emitted. The whole pending block must be
# visible, with the unstable tail appended at the end.


def test_live_line_shows_whole_pending_block():
    """The sentence grows in full, not one word at a time."""
    a = SentenceAssembler()
    a.add_chunk("every narrated animation follows the same", 1.0, 3.0)
    text, t0 = compose_live_line(a, [W("five"), W("stages")])
    assert text == "every narrated animation follows the same five stages", text
    assert t0 == 1.0, t0


def test_live_line_is_block_when_no_tail():
    a = SentenceAssembler()
    a.add_chunk("hello world", 2.0, 2.5)
    text, t0 = compose_live_line(a, [])
    assert text == "hello world", text
    assert t0 == 2.0, t0


def test_live_line_is_tail_when_no_block_yet():
    """Before the first commit there is no block, so the tail carries the line."""
    a = SentenceAssembler()
    text, t0 = compose_live_line(a, [W("hello", 0.4), W("world", 0.6)])
    assert text == "hello world", text
    assert t0 == 0.4, t0


def test_live_line_is_empty_when_nothing_pending():
    a = SentenceAssembler()
    assert compose_live_line(a, []) == ("", 0.0)


def test_live_line_has_no_double_or_dangling_spaces():
    a = SentenceAssembler()
    a.add_chunk("block text", 0.0, 1.0)
    for tail in ([], [W("x")], [W("x"), W("y")]):
        text, _ = compose_live_line(a, tail)
        assert text == text.strip(), repr(text)
        assert "  " not in text, repr(text)
        assert " ," not in text and "( " not in text, repr(text)


def test_live_line_block_and_tail_are_not_duplicated():
    """Block and tail must concatenate, not overlap.

    `draft` previews the block that will later become a `final`, so the
    same text appearing in both is the design -- but the tail must be
    appended exactly once.
    """
    a = SentenceAssembler()
    a.add_chunk("one two three", 0.0, 1.0)
    text, _ = compose_live_line(a, [W("four"), W("five")])
    assert text.count("four") == 1 and text.count("five") == 1, text
    assert text == "one two three four five", text


def test_live_line_tracks_the_block_not_the_tail():
    """After a flush the pending block is gone, so only the tail shows."""
    a = SentenceAssembler()
    a.add_chunk("done", 0.0, 0.5)
    a.flush()
    text, t0 = compose_live_line(a, [W("next", 9.0)])
    assert text == "next", text
    assert t0 == 9.0, t0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok:", t.__name__)
    print("ALL %d PASSED" % len(tests))