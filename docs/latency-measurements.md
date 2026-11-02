# Latency measurements

Raw numbers behind the design decisions in
[`streaming-latency.md`](streaming-latency.md). Machine: Quadro RTX 3000 (6GB),
Xeon E-2276M (6 cores / 12 threads), `small.en` via faster-whisper on CUDA
float16.

## First real noisy-input capture: Zoom lecture (2026-10-23)

Everything prior to this section is measured on **clean narration**. The
Oct 23 09:15 session — a ~51-min lecture watched over Zoom, captured through
WASAPI loopback — is the first measurement against real call-quality audio.
Its artifacts live in `backend/testdata/oct23zoom/` (gitignored; the 108 MB
wav is the fixture). Reference-free only: no ground-truth transcript exists,
so **none of the numbers below is a WER claim**. Analyzer:
`backend/analyze_session.py <stem>`.

Original session, `small.en` cuda/float16, temp-fallback off, 4867 loop steps,
3565 s wall:

| | |
|---|---|
| caption words | 5557 over 51m38s |
| committed-word latency | median **1.83s**, p90 3.03s, max 7.46s |
| 10-min drift of that median | 1.66 → 1.90 → 1.97 → 1.87 → 1.78 → 1.72 — **flat** (adaptive step held) |
| decode pass cost | median 0.66s, p90 0.85s, max **5.0s** |
| pass duty vs wall | 2115s of decode in 3565s = **59%** |
| slow passes ≥1.5s | **12**, 36.2s total, **all on 0.9–4.8s buffers** |
| dup5 within the transcript | 30 (mostly the stock phrases below) |
| hallucination artifacts | 26 — `thank you very much` x19, `thanks for watching` x6, `thanks a lot` x1 |
| polish pass (.md→.txt) | LCS survives 86% / 89% — the LLM rewrites ~13% of the text |

Two headline differences vs the clean-narration numbers above:

1. **Word latency is +0.7s** (1.83s vs 1.10s) but flat for a whole hour on
   noisy audio. Committing requires two agreeing passes, and Zoom audio
   gives the model real disagreements plus pauses — the extra latency is
   LocalAgreement waiting, not backlog.
2. **Pass cost is nearly flat vs buffer** (0.57s @ 1s buffer → 0.82s @ 5s),
   the reverse of the 4s cliff measured on narration — because Zoom audio is
   dense speech, so VAD trims almost nothing, while the large buffers in the
   table belong to cap flushes over *quiet* stretches whose trimmed
   speech is short. Cost follows **speech seconds in the buffer**, not buffer
   seconds.

### Why pass cost varies — measured, not guessed

Four families of cause were in play; three are now measured and one is
absolved:

- **Buffer size / the 4s cliff — mostly absolved for this input.** The cliff
  exists, but on this audio VAD trims little, so the cliff is a gentle slope
  (0.57→0.92s from 1s→5s buffers). The 12s cap-flush rows sit at ~0.59s
  median because those buffers hold quiet audio.
- **Temperature fallback — absolved.** The build runs `--temp-fallback off`;
  the recorded slow passes are not its signature.
- **GPU contention / thermal — absolved (today's runs).** Replaying the
  session's first 2 min repeatedly, `nvidia-smi` sampled every second showed
  SM clocks pinned at 945 MHz / P2 (299 of 308 samples), ≤67°C. No clock dips
  accompany even a 12.9s pass.
- **Content-driven hot spots, magnitude from cuDNN nondeterminism —
  confirmed.** The first 120s of the session audio was replayed through the
  current `RollingStreamer` **four times** (fresh model load each time). Slow
  passes (≥1.5s) reproduce at the **same audio positions** — 23.1/23.4s,
  43.2/43.8s, 57.3s, 63.3/63.6s, 102.3–102.9s, 114.3–114.6s — all on small
  buffers (0.3–1.6s). But the magnitude at each spot swings run to run: 7.7s
  vs 2.6s at 23.4s, 8.1s vs 2.0s at 114.6s, 12.9s at 26.1s in one run where
  the same spot passed in under 2s in two others. Same input, same clock
  state, same model: the variance is the **beam-search instability of
  high-entropy (noisy/hallucination-prone) audio**, consistent with the
  quantified noise floor recorded at the end of
  [streaming-latency.md](streaming-latency.md). The docs' one surviving
  "~4.8s pass under `off`, cause unknown" is the same phenomenon.

### Replay vs the live session (first 2 min, no reference)

| | live session | replay today (`score_stream`, step 0.3) |
|---|---|---|
| wall vs audio | real-time piped | **1.44–1.82x realtime** (slower than live pacing) |
| words in 120s | 191 | 174 |
| mutual LCS overlap | — | **72 words: 38% of live / 41% of replay** |
| artifacts in committed text | 3 in first 2 min | 11 (`thank you` x5, `thanks for watching` x2, `bye` x2) |
| VAD gate | — | skipped 35.2s/120.3s (29%) as non-speech |
| stalls during speech | — | 20 (19 pause/cap flushes by design, 1 deadlock of 1.2s) |

`--dump-audio` + `score_stream.py --dur` on this fixture is now the standing
replay harness for Zoom-quality audio; `probe_live.py` attributes its stalls
without a reference. The duplicate-position hot spots also give the first
concrete pointer for speed-up work: the model slow on the *same* audio spans
they hallucinate on.

### Mitigation sweep on the hot spots (`backend/bench_hotspots.py`)

The six reproducible slow spans (30s total, sliced into
`testdata/oct23zoom/hotslices.wav`) were decoded 8× each per config:

| beam | compute | median | p10–p90 | max |
|---|---|---|---|---|
| 1 | float16 | **1.94s** | 1.83–3.78 | 3.78 |
| 5 | float16 | 2.49s | 2.48–2.66 | 2.66 |
| 1 | int8_float16 | 1.94s | 1.82–2.26 | 2.26 |
| 5 | int8_float16 | 2.82s | 2.78–3.85 | 3.85 |
| 1 | int8 | 1.94s | 1.92–2.10 | 2.10 |
| 5 | int8 | 2.78s | 2.76–2.80 | 2.80 |

Findings:

- **`--compute int8` — rejected.** Identical medians (1.94s), no tail
  taming. Precision is not the driver of hot-spot cost.
- **Beam 1 contradicts the "beam is free" row above** (that sweep was
  measured on narration): on hot-spot content beam 5 pays +0.6–0.9s per
  30s decode. Full 120s replay cost: beam 1 mean pass **0.473s vs 0.543s**,
  wall 191s vs 219s (13% faster). Word counts 168 vs 174; both overlap the
  live session's text within a few points (42%/48% vs 38%/41% LCS) — **inside
  the ±15-point noise floor**, so no accuracy claim either way.
- **The outlier tail is not fixed** by any config (one 3.8–8.2s pass per
  run remains), and within-config 8-run spreads are tight — so the replay's
  1.5→12.9s jumps come from *pass state* (VAD trim and beam path differ per
  cycle), not from cuDNN noise alone.

**Decision: nothing to ship by default.** The pipeline already bounds an
outlier's latency impact (`plan_step` median + 2.5s clamp + `--max-backlog`),
and flipping the beam default on proxy metrics alone is exactly the
Whisper-vs-Whisper mistake recorded at the bottom of
[streaming-latency.md](streaming-latency.md). But `--beam-size 1` is
**already available** at the CLI and is now the *recommended* setting for
Zoom/call-quality input where freezes feel worse: same accuracy within
noise, ~13% cheaper, and greedy decode cannot wander the way the 12.9s pass
suggests beam search can. WER-grade verdict must wait for a human
reference transcript of the fixture.


## Inference cost per pass

**How to measure this correctly.** `WhisperModel.transcribe()` returns a
**lazy generator** — the decoding happens while iterating the segments, not at
the call. Timing the call without consuming it reports ~3ms for work that
really takes ~500ms. Every number below consumes the generator
(`sum(1 for _ in segs)`); `backend/bench_pass.py` does this, and
`test_pass_secs_measures_the_actual_decode` guards it.

Two further traps: faster-whisper pads to `chunk_length=30s`, so cost is much
flatter than buffer length suggests; and cuDNN beam search is not
bit-deterministic, so differences under ~15 WER points are noise.

`transcribe()` wall time, `vad_filter=True`, word timestamps on, median of 5
runs after a discarded warmup (`backend/bench_pass.py`, contiguous audio from
the head of the clip):

| Window | beam 1 | beam 5 | p90 beam 5 | vs a 0.3s step |
|---|---|---|---|---|
| 0.3s | 1.7ms | 1.7ms | 1.9ms | 1% |
| 1.0s | 2.6ms | 2.6ms | 2.8ms | 1% |
| 2.0s | 4.0ms | 4.1ms | 5.8ms | 2% |
| **4.0s** | 463ms | 521ms | **526ms** | **175%** |
| 8.0s | 471ms | 646ms | 648ms | 216% |
| 12.0s | 514ms | 612ms | 615ms | 205% |

Beam size is essentially free — 1 and 5 cost the same. **The step budget, not
the beam, is the lever.**

### In the live pipeline the cliff is much closer

Measured inside the real `feed()` loop over 240s of continuous narration
(801 cycles, wrapper timing each stage):

| stage | median | p90 | max | total |
|---|---|---|---|---|
| VAD (`get_speech_timestamps`) | 2.1ms | 2.9ms | 4.2ms | 0.7s |
| Whisper decode | **500ms** | **1435ms** | **13106ms** | **519.8s** |
| `_cut` | ~0ms | ~0ms | ~0ms | ~0s |

**240s of audio cost 520s of processing — 2.17x real time.** At a fixed 0.3s
step that is unsupportable, and the queue grows without bound. See
[Accumulating latency](#accumulating-latency).

## Model load

| | Time |
|---|---|
| cold (first run after boot) | 3.10s |
| warm | 1.40s |
| first inference after load (CUDA autotune) | 0.31s cold, 0.01s warm |

483 MB of weights on disk (`model.bin`), 563 MB resident. `medium.en` is 1555 MB.

## VRAM

| Model | VRAM |
|---|---|
| `small.en` | 563 MB (9% of 6 GB) |
| `medium.en` | 1555 MB (25%) |

Both fit comfortably. A GPU-resident correction model would contend with
Whisper, which is why the LLM pass runs on CPU.

## Accumulating latency

The symptom: captions lag a little at first, and the lag grows the longer the
session runs.

**Cause.** One pass over ~0.5s of real speech costs ~500ms (median, measured
above) while the step was fixed at 300ms. Every cycle therefore took longer
than the audio it consumed, so audio queued faster than it drained and never
recovered:

```
240s of audio cost 520s of processing  ->  drift +270s after 4 minutes
```

Measured end-to-end latency before the fix, real-time paced:

| audio position | median latency |
|---|---|
| 0–30s | 28.9s |
| 60–90s | 106.8s |
| 120–150s | 152.2s |
| 210–240s | **271.2s** |

**It was worse with continuous narration** than with the psql capture, because
music is skipped by the VAD gate while every step of real speech pays the full
pass cost.

**Fix — adaptive step plus a drift guard** (`plan_step` in `captions.py`):

- The step is sized from the **measured p90 pass time** of recent cycles
  (`--step-secs` is now a *floor*, `--step-headroom` the multiplier), so the
  pipeline consumes audio at exactly the rate the machine can process it.
- `--max-backlog` (default 1.5s) caps queued audio; anything older is dropped
  rather than allowed to accumulate. This makes unbounded lag impossible even
  if inference degrades further.

| | before | after |
|---|---|---|
| processing vs real time | 2.17x too slow | **1.54x faster** |
| cycles overrunning | 86% | **8%** |
| median latency | growing, 29s → 271s | **1.10s, flat** |
| p90 / max latency | 253s / 276s | **3.09s / 5.07s** |

The result: the adaptive step reported ~0.9s (measured median), which is what
the larger step is for.

### This fixed the accuracy too

A larger step gives the model more context, and the earlier step sweep already
showed accuracy improving as the step grows (smaller steps are worse because
every commit resets the buffer). At the measured ~0.9s step:

| | step 0.3 | **step 0.9** | old fixed 1.5s |
|---|---|---|---|
| testclip WER | 20.2% | **9.6%** | 32.3% |
| testclip survival | 91.3% | **92.3%** | — |
| genesis-4 WER | 47.9% | **32.4%** | 43.8% |
| genesis-4 survival | 58.5% | **69.6%** | 62.4% |
| dup5 | 0 | 0 / 2 | 0 / 1 |

The latency floor bought by correctness turned out to cost nothing in accuracy
— it gained it.

## LLM correction pass

`Qwen2.5-0.5B-Instruct` Q4 GGUF (469 MB) via llama.cpp, CPU, 6 threads.

| | Time |
|---|---|
| model load | 0.49s |
| per 200-char line | 1.19–1.69s |

Runs on a worker thread and only on committed sentences, so it is off the live
path. A 200-char line is roughly 4.4s of speech at the measured ~9 words/sec,
so the correction adds 27–38% to the time for a line to become permanent —
but the live line is unaffected.

The backend sets `OMP_NUM_THREADS=1` for the resampler, and llama.cpp honours
that variable, so `--correct-threads` must be set explicitly or the model runs
single-threaded.

## Historical: fixed non-overlapping windows vs accuracy

This sweep drove the *previous* design (each window decoded independently) and
is kept for the record. It is superseded by the rolling streamer, whose
accuracy is tuned by `--step-secs` instead — see
[step size](#accumulating-latency) above.

WER against the Genesis 4 reference, fixed non-overlapping windows:

| Window | WER | dup5 | Latency |
|---|---|---|---|
| 1.0s | 69.8% | — | 1.0s |
| 1.5s | 58.6% | 6 | 1.5s |
| 2.0s | 34.3% | 0 | 2.0s |
| 2.5s | 32.8% | 5 | 2.5s |
| 3.0s | 29.6% | 0 | 3.0s |
| 5.0s | 18.8% | 1 | 5.0s |
| 6.0s | 17.1% | 1 | 6.0s |
| whole clip, one call | 20.6% | 1 | n/a |

Substitutions dominate the error at short windows (202 of ~350 errors at 1.5s),
not deletions or duplication — the model mishears because it lacks context,
which is precisely what a rolling buffer supplies.