# Latency measurements

Raw numbers behind the design decisions in
[`streaming-latency.md`](streaming-latency.md). Machine: Quadro RTX 3000 (6GB),
Xeon E-2276M (6 cores / 12 threads), `small.en` via faster-whisper on CUDA
float16.

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