"""LLM correction pass for completed caption sentences.

Runs a small instruct model on its own thread so transcription never blocks.
Each committed sentence is enqueued and polished; the result is emitted as a
`corrected` event carrying the block id, which the TUI uses to patch the
already-displayed line.

Design notes:

- **Only completed sentences.** `submit()` is called from the assembler's
  commit path, never for drafts, so the live line is never rewritten under
  the reader.
- **Bounded queue.** If the model cannot keep up, stale items are dropped
  rather than queued indefinitely: a correction that lands many blocks late
  would patch a line the reader has moved past, which is worse than no
  correction at all.
- **Explicit thread count.** The backend sets `OMP_NUM_THREADS=1` so the
  resampler does not oversubscribe the CPU, and llama.cpp honours that
  variable. Without an explicit `n_threads` the model would run
  single-threaded on a 12-thread machine, several times slower than needed.
- **Graceful degradation.** Any failure leaves the raw line in place. A
  missing model or an inference error must never take the captions down.
"""

import json
import os
import queue
import sys
import threading

# Corrections more than this many blocks behind the newest emitted block are
# dropped: the reader has scrolled past them.
MAX_STALE_BLOCKS = 2

SYSTEM_PROMPT = (
    "You repair speech-to-text transcripts of narrated audio.\n"
    "\n"
    "The words were guessed from sound and are often wrong. Reconstruct what "
    "the speaker most likely actually said, using the sentence as a whole to "
    "decide. If a phrase is not idiomatic English, or does not fit what is "
    "being discussed, replace it with the most plausible real phrasing.\n"
    "\n"
    "Rules:\n"
    "- Prefer the reading that makes the sentence make sense as a whole, even "
    "if it means changing several words.\n"
    "- Keep names, places and numbers as they were heard unless they are "
    "obviously not real.\n"
    "- Do not invent new events, facts or sentences that were not there.\n"
    "- Do not summarise or shorten. The result should be roughly the same "
    "length as the input.\n"
    "- Always output a complete, properly punctuated sentence or two.\n"
    "\n"
    "Reply with the corrected transcript only. No preamble, no quotes."
)


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


class Corrector:
    """Background LLM polisher for committed sentences."""

    def __init__(self, model_path, threads=0, max_tokens=160, n_ctx=512):
        self.model_path = model_path
        self.threads = threads if threads > 0 else (os.cpu_count() or 4)
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.q = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.thread = None
        self.llm = None
        self.newest_id = 0        # highest block id submitted so far
        self.dropped_stale = 0
        self.failures = 0

    # -- lifecycle -------------------------------------------------------

    def start(self):
        """Load the model on the worker thread and begin processing."""
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="corrector")
        self.thread.start()

    def stop(self, timeout=5.0):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def submit(self, block_id, text):
        """Queue a completed sentence. Never blocks the caller."""
        self.newest_id = block_id
        try:
            self.q.put_nowait((block_id, text))
        except queue.Full:
            # Drop the oldest so the newest text is the one being worked on;
            # a stale correction is useless to the reader.
            try:
                self.q.get_nowait()
                self.q.put_nowait((block_id, text))
            except (queue.Empty, queue.Full):
                pass

    # -- worker ----------------------------------------------------------

    def _run(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            _log("correction disabled: llama_cpp not installed")
            return
        try:
            self.llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx,
                             n_threads=self.threads, verbose=False)
            _log("corrector ready (%d threads)" % self.threads)
        except Exception as e:  # noqa: BLE001 — never fatal
            _log("correction disabled: model load failed: %s" % e)
            return

        while not self.stop_event.is_set():
            try:
                item = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            block_id, text = item
            if self.newest_id - block_id > MAX_STALE_BLOCKS:
                self.dropped_stale += 1
                continue
            corrected = self._correct(text)
            if corrected:
                self._emit(block_id, corrected)

    def _correct(self, text):
        if self.llm is None:
            return None
        try:
            r = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": text}],
                max_tokens=self.max_tokens, temperature=0.0)
            out = r["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001 — never fatal
            self.failures += 1
            _log("correction failed: %s" % e)
            return None
        # A model that returns nothing, or echoes nothing usable, must not
        # blank out a line the reader is looking at.
        if not out or len(out) < 2:
            return None
        return out

    @staticmethod
    def _emit(block_id, text):
        sys.stdout.write(json.dumps(
            {"type": "corrected", "id": block_id, "text": text},
            ensure_ascii=False) + "\n")
        sys.stdout.flush()