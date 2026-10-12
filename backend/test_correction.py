"""Unit tests for the correction worker: queueing, staleness, degradation.

No model is loaded — a stub stands in for the LLM so these run instantly and
without a 500MB download.

Run: ..\\.venv\\Scripts\\python.exe backend\\test_correction.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import correction


class StubLlama:
    def __init__(self, reply="CORRECTED", delay=0.0, fail=False):
        self.reply = reply
        self.delay = delay
        self.fail = fail
        self.seen = []

    def create_chat_completion(self, messages, max_tokens, temperature):
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("stub failure")
        self.seen.append(messages[-1]["content"])
        return {"choices": [{"message": {"content": self.reply}}]}


def make(stub, threads=2):
    c = correction.Corrector("unused", threads=threads)
    c.llm = stub
    return c


def capture_emits(fn):
    """Run fn with stdout captured; return the emitted events."""
    import io
    import json

    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    try:
        fn()
    finally:
        sys.stdout = real
    out = []
    for line in buf.getvalue().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def test_submit_never_blocks_when_full():
    c = make(StubLlama(delay=0.05))
    t0 = time.time()
    for i in range(20):          # queue capacity is 4
        c.submit(i, "sentence %d" % i)
    elapsed = time.time() - t0
    assert elapsed < 0.5, "submit blocked for %.2fs" % elapsed
    assert c.q.qsize() <= 4, c.q.qsize()


def test_stale_corrections_are_dropped():
    c = make(StubLlama(reply="fixed"))
    c.submit(1, "old")
    c.newest_id = 99             # far ahead: block 1 is now stale
    c.q.put((1, "old"))
    events = capture_emits(lambda: _drain(c))
    assert events == [], events
    assert c.dropped_stale >= 1


def test_fresh_correction_is_emitted():
    c = make(StubLlama(reply="fixed text"))
    c.submit(7, "raw text")
    events = capture_emits(lambda: _drain(c))
    assert len(events) == 1, events
    assert events[0]["type"] == "corrected"
    assert events[0]["id"] == 7
    assert events[0]["text"] == "fixed text"


def test_llm_error_leaves_line_untouched():
    c = make(StubLlama(fail=True))
    c.submit(1, "raw")
    events = capture_emits(lambda: _drain(c))
    assert events == [], "a failed correction must emit nothing"
    assert c.failures == 1


def test_empty_reply_is_rejected():
    """A model returning nothing must not blank out a displayed line."""
    for reply in ("", " "):
        c = make(StubLlama(reply=reply))
        c.submit(1, "raw")
        events = capture_emits(lambda: _drain(c))
        assert events == [], (reply, events)


def _drain(c):
    """Run one worker iteration over whatever is queued."""
    item = c.q.get_nowait()
    if item is None:
        return
    block_id, text = item
    if c.newest_id - block_id > correction.MAX_STALE_BLOCKS:
        c.dropped_stale += 1
        return
    corrected = c._correct(text)
    if corrected:
        c._emit(block_id, corrected)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok:", t.__name__)
    print("ALL %d PASSED" % len(tests))