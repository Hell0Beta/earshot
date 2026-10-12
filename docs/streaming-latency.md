# Streaming latency: what was tried, what worked, what didn't

This is the record of overlap handling and caption latency in earshot. Every
number here was measured on this machine (Quadro RTX 3000, Xeon E-2276M, 6
cores / 12 threads, `small.en` on CUDA) against a real reference.

It exists so that none of the failed approaches are re-attempted from memory.
The short version: **four different ways of detecting overlapping audio were
built, measured, and reverted.** The lesson is at the bottom.

## Ground truth

Two clips ship with a transcript of what was actually spoken:

| Clip | Audio | Reference |
|---|---|---|
| `001_genesis-4` | `voice/narration.wav`, 240s | `transcript/narration.srt`, 597 words |
| `000-testclip` | `voice/narration.wav`, 41s | `transcript/narration.srt`, 96 words |

Both are clean narration by the same speaker. They isolate pipeline behaviour
from audio quality, which is what a dedup harness needs — see the caveat at
the end.

Metrics, reported **separately and never blended**:

- **WER** — word error rate vs the reference (substitutions + deletions + insertions)
- **SURVIVAL** — fraction of reference words present in the output, in order.
  Catches deleted speech, which WER alone reports only as a diffuse cost.
- **dup5** — repeated 5-word phrases. Catches re-appended overlap.

Blending them is what let a transcript-eating bug hide: `_collapse_intra`
*lowered* dup5 while *deleting* words, so any single score improved while the
transcript got worse.

## Baselines

| Design | WER (testclip) | dup5 |
|---|---|---|
| Whole clip in one `transcribe()` call | 20.6% | 1 |
| Pre-session build (v1.2.1, 1.5s window + 0.2s overlap) | — | 0 |
| Sliding window, `--context-secs 3.0 --new-secs 0.5` | **167.5%** | 16 |
| Fixed 1.5s windows, no overlap | 32.3% | 0 |
| **Rolling + stable prefix, step 0.3** (shipped) | **20.2%** | **0** |

The sliding window was a **2.5× accuracy regression** against the build it
replaced (67.2% WER on `genesis-4`), and every metric in use at the time
reported it as healthy.

The shipped design matches the whole-clip-in-one-call baseline for accuracy
(20.2% vs 20.6%) while running at 0.36s median latency instead of offline.

## What was tried

### 1. Exact-match overlap trim
Drop the leading words of a chunk that exactly matched the emitted tail.

**Failed.** The same audio re-decoded from a different window boundary comes
back reworded or re-punctuated (`wild-` vs `wild`, `' You'` vs `You`), so
exact matching missed most of the restatement and re-appended the whole
prefix. This is the classic LocalAgreement problem solved wrong.

### 2. Longest-prefix fuzzy match
Score prefix words against the tail with an edit distance, allow a mismatch
budget proportional to run length.

**Worked at a relaxed cadence, failed as the cadence tightened.** Duplicated
6-grams went 0 → 6 → 14 as `--new-secs` went 0.75 → 0.35 → 0.20. The cause is
structural: the cut point is the *longest self-consistent prefix*, and that
prefix shrinks as the restatement grows. Latency and correctness were fighting
over the same knob.

### 3. Edit-distance alignment (Viterbi-style DP)
Replace the greedy prefix scan with a real alignment, so one reworded word
costs one edit instead of invalidating the whole prefix.

**Measured, not adopted.** Against the same reference: duplicated 6-grams 6 → 1
at `new 0.35` (a real win) but no better at the shipped cadence, and *worse*
divergence there (23.1% vs 16.5%). Kept in the git history rather than shipped.

### 4. Word-timestamp gating
Turn on `word_timestamps=True` and drop any word whose absolute audio time was
already emitted. Intended to replace all text matching with arithmetic.

**Duplication went to zero at every cadence — and accuracy got worse.** Two
causes:

- Whisper's word timings jitter 0.03–0.05s between decodes. Across two window
  boundaries only **2 of 11 words** landed on identical timestamps.
- Adjacent words sit closer together than that jitter, so the gate could not
  separate "same word re-decoded" from "next word".

It also under-emitted badly (172 words vs 357 for the same audio), i.e. it
deleted real speech. Reverted.

**Segment** start times, by contrast, *are* stable — VAD trims leading silence,
so the first segment's start tracks the window boundary exactly. A
segment-level version of this idea was never built.

### 5. `_collapse_intra`
Collapse any n-gram that repeats twice within a single pass, on the theory
that Whisper loops on a phrase.

**Destroyed real speech.** Recited text repeats constantly — Genesis 4 alone
contains `was the father of` five times, `the lord said to` three times — and
the function deleted *everything between* the two copies:

```
in : 'and he said unto her and he said unto her'
out: 'and he said unto her'                     5 words destroyed

in : 'The LORD God said and the LORD God said unto the woman'
out: 'the LORD God said unto the woman'         5 words destroyed
```

No metric caught it because it *reduced* dup5 while deleting words. It shipped
for several iterations before a reference transcript exposed it. Its test suite
even encoded the bug as intended behaviour, which is why deleting it required
deleting the test too.

### 6. Fixed non-overlapping windows
Drop the overlap entirely: hop == window. Nothing re-reads anything, so there
is nothing to deduplicate and the whole failure class disappears.

**Shipped.** 32.3% WER on testclip with dup5 = 0, against 167.5% / 16 for the
sliding window. The cost is that latency equals the window length.

### 7. LLM correction pass
`Qwen2.5-0.5B-Instruct` (Q4 GGUF, `llama-cpp-python`, CPU) polishes each
*committed* sentence on a worker thread; the TUI patches the line in place and
the transcript is rewritten in place. Off by default.

**Readability gain, modest accuracy gain.** testclip: 32.3% → 30.2% WER at
~1.2s per 200-character line. Genesis 4: 63.5% → 65.3% (worse) and dup5 0 → 11.
It reliably fixes punctuation and capitalisation, and occasionally a broken
token (`every narrated anim` → `every narrated animation`), but does not
reconstruct mishearings — `tie medal card` for `title card` survives untouched.
A 0.5B model is too weak for that; the context needed to spot a mishearing is
exactly what a small model lacks.

### 8. Rolling buffer with stable-prefix commits (shipped)
Keep one growing audio buffer since the last commit; every `--step-secs`
re-decode it; commit only the words **two consecutive hypotheses agree on**
(LocalAgreement-2). The rest is the unstable tail, shown live and free to
change. A pause or a buffer cap flushes it.

**Shipped.** It sidesteps the whole failure class rather than repairing it:
nothing asks "where does the old text end?", because agreement between
passes over the same buffer decides what is new.

Measured with `backend/score_stream.py` (same scorer for both columns):

| testclip (41s) | fixed 1.5s | rolling 0.3/0.2 |
|---|---|---|
| WER | 32.3% | 20.2% |
| survival | — | 91.3% |
| dup5 | 0 | **0** |
| median word latency | 1.5s | 0.36s |

> Superseded — those numbers were taken at the old fixed 0.3s step, which was
> not sustainable (see
> [Accumulating latency](latency-measurements.md#accumulating-latency)). At
> the adaptive step the shipped defaults score **9.6% WER on testclip** and
> **32.4% on genesis-4**, both better than the 32.3% / 43.8% fixed-window
> design, at a flat ~1.1s median latency.

| genesis-4 (240s) | fixed 1.5s | rolling 0.3/0.2 |
|---|---|---|
| WER | **43.8%** | 47.9% |
| survival | **62.4%** | 58.5% |
| dup5 | 1 | **0** |

Four things this established:

- **Duplication is structurally impossible.** dup5 = 0 on both clips.
- **Latency is 4x lower.** Median 0.36s (1.2 steps), p90 0.78s, vs 1.5s
  fixed. The floor is 1-2 steps, as the theory predicts.
- **On ordinary vocabulary it is clearly more accurate** (20.2% vs 32.3%).
- **Agreement is doing the work, not the heuristics.** With `--pause-secs 0`
  every pass flushes unconditionally — i.e. the old non-overlapping
  behaviour — and WER collapses to **102.9%**. Pauses only decide *when* to
  stop waiting.

#### Faster steps make it *worse*

The opposite of the design assumption, and worth recording:

| step | WER (testclip) | dup5 | wall vs realtime |
|---|---|---|---|
| 0.3s | **20-23%** | 0 | 1.7x |
| 0.2s | 27.9% | 0 | 3.9x |
| 0.1s | 45.2% | 2 | 7.9x |

Cause: **every commit resets the buffer to near zero** (audio behind the last
committed word is dropped). At a 0.1s step, commits arrive fast enough to keep
the buffer permanently short, reinstating exactly the small-window problem a
rolling buffer exists to solve. Latency and accuracy run *opposite* to the
intuition here.

#### Where the missing words go

`backend/diagnose_loss.py` attributes every lost reference word to either the
streamer or the model:

| genesis-4 | words | share |
|---|---|---|
| **NEVER_DECODED** (model never emitted them) | **132** | **20.3%** |
| LOST_IN_STREAMER (decoded, never committed — our bug) | 15 | **2.3%** |

testclip: 8 never decoded (7.7%), **1 lost in streamer (1.0%)**.

So the streamer costs ~1-2% in exchange for zero duplication and 4x lower
latency. Everything else is `small.en` not knowing Biblical proper nouns —
`abel`, `adam`, `eve`, `seth`, `zillah`, `adah`, `irad`, `mehujael`,
`methushael`, `genesis`, `firstborn` never appear in **any** pass. No windowing
scheme recovers a word the decoder never produces.

That is also why genesis-4 WER stays high for *both* designs: the clip is
adversarial for a general-purpose model, not for this architecture.

#### medium.en was tried and rejected

| genesis-4 | rolling / small.en | rolling / medium.en |
|---|---|---|
| WER | **47.9%** | 51.6% |
| survival | 58.5% | 58.8% |
| wall | 469s | 1300s |

`medium.en` is worse **and** 2.8x slower, and it hallucinated conversational
filler absent from the audio (`E We'll see you next week`, `Thank you`,
`Bye!`, `Go`). A larger language model fills gaps from its priors on continuous
unbroken narration. Keep `small.en`.

#### Remaining known issues

- **Streamer loss ~2.3%** on genesis-4 (words decoded but never agreed on).
  The price of requiring two passes.
- **43 discard events** from the boundary filter over 1670 passes — words
  starting within `tolerance` of the last commit. Low, but non-zero.
- **Proper nouns outside the model's vocabulary are unrecoverable.** Only a
  better or domain-adapted model, or `--prompt-chars` seeding known names,
  can help.

## Why the first four failed

Overlap can only be repaired by *recognising the restatement*. Whisper
re-decodes the same audio differently from every window boundary, so the
restatement is never verbatim and every heuristic that tried to match it had a
cadence beyond which it lost.

All four heuristics attacked the same question — "where does the old text end?"
— and all four were answering it from text, which is the noisiest available
signal.

## What replaced them

The shipped design asks a different question: *did the model say this again on
the next pass?* Comparing the current hypothesis to the previous one over the
same buffer commits the agreed prefix and redraws everything after it. No
similarity threshold, no anchor window, no edit-distance budget — nothing to
retune when the cadence changes, which is what sank every previous attempt.

The remaining accuracy ceiling is no longer overlap handling at all. It is the
model's vocabulary: on genesis-4, 20.3% of reference words are never decoded in
any pass.

## Non-speech hallucination and caption stalls

A different failure class, found by testing a real capture
(`Videos/Psql Fix(2).mp4`, 246s of screen recording: ~100s of narration over
~145s of background music). Measured with `backend/probe_live.py`.

### Symptom

`Thank you for watching` appearing mid-session, and captions freezing for
several seconds afterwards.

### Root cause — one bug, two symptoms

1. **Whisper hallucinates on non-speech.** Given a 15s stretch of music it
   emits `"Thanks for watching!"`. Silero VAD correctly reports **0.0s** of
   speech for that same stretch, versus ~85% of the buffer for real
   narration.
2. **The phantom then deadlocked output.** A hallucination, by definition,
   never agrees across two passes, so it sits at the head of the unstable
   tail. `StablePrefix.update` commits a *longest common prefix*, so a
   disagreeing head blocks **every** commit — not just that word. With no
   commit, `last_end` never advances, `_cut` never trims, and the buffer
   grows until `max_buf_secs` (12s) force-flushes.

That is the "it rebuilt the initial sample" symptom: not a slow start, a
deadlock lasting up to 12s.

Three fixes, all in `backend/streaming.py`:

- **VAD gate** — never call the model on a buffer Silero says has no speech;
  discard any pending tail (it was decoded from that non-speech). Bounded to
  buffers >= `--gate-min-buf-secs` so short speech is never dropped.
- **Watchdog** (`--stall-passes`) — force-flush after N passes with no
  commit, so a stall is bounded by a few steps rather than the buffer cap.
- **Empty-hypothesis discard** — when `transcribe()` returns nothing, drop
  the tail instead of holding unconfirmed words whose audio has moved on.

### Measured, same file, same scorer

| | pre-fix | post-fix |
|---|---|---|
| artifacts | 10 | **5** |
| deadlocks | 8 (max **9.30s**) | **2 (max 1.50s)** |
| speech-gap median | 0.60s | 0.60s |
| inference per pass | 0.0254s | **0.0033s** |
| audio correctly skipped as non-speech | 0s | **156.6s** |

Two intermediate attempts made it worse and were reverted, both measured:

| attempt | artifacts | words | why it failed |
|---|---|---|---|
| gate at any buffer size | 7 | 194 | a fresh 0.3s step of speech often reports <0.1s speech, so real speech was discarded |
| gate at 0.6s | 8 | 209 | 0.6s is inside the speech+music transition; artifacts went back up and deadlock max hit 18s |

The working configuration requires the buffer to be allowed to *grow* when
`transcribe()` returns nothing, so the gate can classify the run. Clearing the
buffer on every empty hypothesis keeps it tiny, the gate never engages, and
Whisper keeps being handed slivers of music.

### Also fixed

- `hallucination_silence_threshold` had silently defaulted to `None` (**off**);
  it now reaches the model. Five other CLI flags parsed but were never passed
  to `transcribe()`. `test_confidence_gates_reach_the_model` guards this.
- A literal `%` in a help string broke `--help` (and `earshot doctor`)
  entirely. `test_help_text_renders` guards this.

## Backend crash on odd-length audio buffers

Symptom: the backend died roughly a minute into a session, with no traceback,
just `ERROR: buffer size must be a multiple of element size` in
`.backend.log`. Intermittent, and only after the adaptive step landed.

Not silence, not elapsed time, and not the drift guard — every crashing log
showed `backlog 0.00s`. The cause was a float step:

1. `plan_step` returns a **float** target (e.g. `0.9149998`), so
   `want = int(bytes_per_sec * target)` is often **odd**.
2. `collect_step` zero-pads the shortfall up to that odd length.
3. `bytes_to_mono16k` calls `np.frombuffer(raw, dtype=np.int16)`, which
   requires a whole number of samples and raises.

The old fixed `--step-secs 0.3` multiplied out to an even byte count every
time, so this could only appear once the step became adaptive.

Two fixes, defence in depth:

- `collect_step` snaps the step down to a whole number of device frames
  (`channels * 2` bytes), making an unaligned buffer structurally impossible.
- `bytes_to_mono16k` truncates to an even length before `frombuffer`, so any
  future caller is safe.

Measured on real audio through the previously-crashing path: **32 of 300
steps (11%) required the snap** and would previously have raised; the run
completed cleanly. `test_step_bytes_are_always_whole_frames` and
`test_bytes_to_mono16k_survives_odd_length` guard both halves — reverting the
fix reproduces the exact production `ValueError`.

Also fixed: the top-level handler printed only `str(e)`, which is why the
crash had no traceback. It now prints a full `traceback.print_exc()` to
stderr, which the Go frontend copies into `.backend.log`.

## The metrics lesson

Every metric used during the sliding-window work compared **Whisper's output
against Whisper's output** — a one-shot transcription of the same audio as the
"reference". That comparison cannot detect a Whisper-shaped error, so a change
that deleted real speech while duplicating less looked like a strict
improvement. It reported the regression as healthy for several iterations.

The regression was found only once a real transcript of what was spoken existed
(`001_genesis-4/transcript/narration.srt`). **Any future change to the
transcription path should be scored against that.**

Two further traps worth recording:

- **Whole-second log timestamps.** The backend logged `[%H:%M:%S]`, which
  quantises every reading to 1.00s. A real 0.795s loop period measured as
  "1.00s". Latency work needs `perf_counter`.
- **Lazy generators measure as free.** `WhisperModel.transcribe()` returns a
  generator; the decode happens while iterating it. Timing the call alone
  reported ~3ms for work that truly takes ~500ms. This produced two rounds of
  confidently wrong latency numbers before it was caught, and it also hid the
  real cause of the accumulating-lag bug. `backend/bench_pass.py` exists so
  the mistake is not repeated.
- **Run-to-run variance.** cuDNN beam search is not bit-deterministic: the same
  config scored WER 81.2% and 67.7% on two runs. Differences under ~15 points
  are noise; single-run comparisons are meaningless.

## Caveat on all of this

Every number here comes from **clean narration**. It isolates pipeline
behaviour, which is what a dedup harness is for, but it says nothing about
noisy call audio, which is where earshot actually gets used. The
`--correct-model` pass and the window-size tradeoffs are both unmeasured
against real noisy input.