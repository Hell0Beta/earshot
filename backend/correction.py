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
- **One worker, two jobs.** Per-sentence repair and whole-document polish share
  the same `Llama` instance, because llama.cpp contexts are not safe to call
  from two threads at once. Polish requests are served first: they are
  user-initiated (copy, mode switch) and waiting on a stale sentence repair
  would be visible.
"""

import json
import os
import queue
import re
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

# Voice-to-text prompt. The model is a 0.5B instruct model, and a small model
# asked to "rewrite" summarises instead: measured against a 42-word dictation it
# returned a 12-word summary, dropping the setup entirely. So its job is
# deliberately narrow — punctuation and capitalisation only — and the fillers it
# cannot be trusted to remove are removed deterministically first. A one-shot
# example is used instead of more rules: small models follow patterns far more
# reliably than instructions.
POLISH_PROMPT = (
    "You punctuate dictated text. Change nothing else.\n"
    "\n"
    "Example:\n"
    "Input: so the worker restarted twice last night and i think the queue backed up\n"
    "Output: So the worker restarted twice last night, and I think the queue backed up.\n"
    "\n"
    "Add punctuation and capitalisation only. Do not add, remove or reorder any "
    "words. Output the punctuated text only: no introduction, no markdown."
)

# Disfluency that is never content. "like" and "i mean" are deliberately absent:
# both are ordinary English in plenty of sentences, and deleting them changes
# meaning. "you know" is removed except after do/did/does, where it is a real
# question ("do you know how this works"). Each lookbehind is fixed width, as
# Python's re requires, and they stack.
STUTTER_RE = re.compile(
    r"\b(?:um+|uh+|uhm+|erm+|er+|ah+|hmm+|mhm+|mm+)\b,?\s*",
    re.IGNORECASE)
YOU_KNOW_RE = re.compile(
    r"(?<!do )(?<!did )(?<!does )(?<!don't )(?<!didn't )"
    r"\byou know\b[:,]?\s*",
    re.IGNORECASE)

# "the the" / "is is": immediate repeats are always a stutter.
REPEAT_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)


def clean_fillers(text):
    """Strip disfluency before the model sees the text.

    Deterministic because a model this small does not reliably follow a
    "remove filler" instruction, and a missed filler is the most visible sign
    that the text was dictated.
    """
    s = YOU_KNOW_RE.sub(" ", text)
    s = STUTTER_RE.sub("", s)
    s = REPEAT_RE.sub(r"\1", s)
    s = re.sub(r"^[,;\s]+", "", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s.strip()


# A polish that loses more than this share of the input words has summarised
# instead of punctuated. Reject it and leave the raw text up: a readable
# transcript with a whole idea missing is worse than a rough one.
POLISH_MIN_KEEP = 0.8

# A dictation longer than this is truncated from the start before polishing, so
# the request always fits the model's context. Roughly 1500 words: far beyond a
# typical dictated prompt.
POLISH_MAX_CHARS = 6000


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


class Corrector:
    """Background LLM polisher for committed sentences."""

    def __init__(self, model_path, threads=0, max_tokens=160, n_ctx=2048):
        self.model_path = model_path
        self.threads = threads if threads > 0 else (os.cpu_count() or 4)
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.q = queue.Queue(maxsize=4)
        # One slot: a newer polish request always wins, because the reader is
        # waiting on the most recent text, not the one they replaced.
        self.polish_q = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread = None
        self.llm = None
        self.newest_id = 0        # highest block id submitted so far
        self.dropped_stale = 0
        self.failures = 0
        self.polished = 0
        self.polish_failures = 0

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

    def submit_polish(self, text):
        """Queue a whole-document polish. Never blocks the caller.

        Replaces any pending polish: the document only grows, so the newest
        request already contains the previous one's text.
        """
        if not text or not text.strip():
            return
        try:
            self.polish_q.put_nowait(text)
        except queue.Full:
            try:
                self.polish_q.get_nowait()
                self.polish_q.put_nowait(text)
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
            # Polish first: a reader is waiting on this one.
            try:
                doc = self.polish_q.get_nowait()
            except queue.Empty:
                doc = None
            if doc is not None:
                self._serve_polish(doc)
                continue

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

    def _polish(self, text):
        """Punctuate a whole document. None on any failure."""
        if self.llm is None:
            return None
        doc = clean_fillers(text.strip())
        if not doc:
            return None
        if len(doc) > POLISH_MAX_CHARS:
            # Keep the tail: a long dictation is more likely to continue than
            # to be re-read from the start, and the head is already on screen.
            doc = "…" + doc[-POLISH_MAX_CHARS:]
        before = len(doc.split())
        try:
            r = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": POLISH_PROMPT},
                          {"role": "user", "content": doc}],
                max_tokens=min(1024, max(64, len(doc) // 2 + 32)),
                temperature=0.0, repeat_penalty=1.15,
                stop=["\n\n---", "\n---", "\n**", "\nInput:", "\nOutput:"])
            out = r["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001 — never fatal
            self.polish_failures += 1
            _log("polish failed: %s" % e)
            return None
        # Same rule as corrections: an empty reply must not blank the pane.
        if not out or len(out) < 2:
            return None
        # A small model sometimes summarises instead of punctuating. Losing a
        # whole idea is worse than leaving the raw text up.
        if before >= 10 and len(out.split()) < before * POLISH_MIN_KEEP:
            self.polish_failures += 1
            _log("polish rejected: %d words in, %d out (summarised?)"
                 % (before, len(out.split())))
            return None
        return out

    def _serve_polish(self, doc):
        """Answer a polish request, even when it failed.

        An empty `polished` event is the "nothing changed" signal: the TUI
        clears its in-flight flag and keeps the text it already has. Staying
        silent would leave the pane saying "polishing…" forever.
        """
        out = self._polish(doc)
        if out:
            self.polished += 1
        self._emit_polished(out or "")

    @staticmethod
    def _emit_polished(text):
        sys.stdout.write(json.dumps(
            {"type": "polished", "text": text}, ensure_ascii=False) + "\n")
        sys.stdout.flush()