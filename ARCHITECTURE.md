# Earshot Architecture

## Components

```
┌─────────────────┐  WASAPI loopback   ┌──────────────┐  JSON-lines   ┌─────────────┐
│  Any app audio  │ ─────────────────▶ │ Python       │ ───────────▶ │ Go Bubble   │ ──▶ screen
│  (Teams/Zoom/…) │   48kHz stereo     │ backend      │  stdout       │ Tea TUI     │    + .md file
└─────────────────┘                    └──────────────┘               └─────────────┘
                                              │                             │
                                              ▼                             ▼
                                       D:\earshot\models            transcripts/*.md
```

## Python backend (`backend/`)

### Audio pipeline

> **M1 finding (verified 2026-09-26):** the Realtek WASAPI loopback delivers
> **zero callbacks while the render stream is idle** — not even silence. The
> backend chunk timer MUST be wall-clock based (queue reads with timeout),
> treating missing data as digital silence. Never gate progress on callback
> counts, or the pipeline stalls whenever the user isn't playing audio.

1. **Capture** — `pyaudiowpatch`, `get_default_wasapi_loopback()`, callback → thread-safe queue. Flags: `--list-devices`, `--device-index`.
2. **Convert** — stereo → mono (`numpy` mean), resample 44.1/48kHz → 16kHz (`scipy.signal.resample_poly`), float32 in `[-1, 1]`.
3. **Rolling buffer + stable-prefix commit** (`backend/streaming.py`) — one
   growing audio buffer since the last commit. Each cycle re-decodes the whole
   buffer with word timestamps, and a word is committed only once **two
   consecutive hypotheses agree** on it (LocalAgreement-2). Committed words go
   to the assembler; the rest is the unstable tail, shown live and free to
   change. A pause (`--pause-secs`, 0.2) or a buffer cap (`--max-buf-secs`, 12)
   flushes the tail.
   - **The step is adaptive, not fixed.** `--step-secs` is a *floor*; each
     cycle is sized from the measured p90 pass time of recent cycles
     (`plan_step`). One pass over ~0.5s of real speech costs ~500ms on this
     GPU, so a fixed 0.3s step could not keep up: every cycle overran the
     audio it consumed and caption lag grew without bound (29s → 271s over
     four minutes). `--max-backlog` (1.5s) additionally drops stale audio, so
     unbounded lag is impossible even if inference degrades.
   - Measured after the fix: latency is **flat at ~1.1s median** (was growing
     to 271s), and accuracy *improved* — 9.6% WER on testclip, 32.4% on
     genesis-4, both better than the 32.3% / 43.8% of the old fixed-window
     design.
   - This replaced four text-based overlap heuristics, all reverted — see
     [`docs/streaming-latency.md`](docs/streaming-latency.md) for the full
     record and why each failed. The short version: every one of them asked
     *"where does the old text end?"*, and Whisper re-decodes the same audio
     differently from every boundary, so the restatement is never verbatim.
     This design asks a different question — *did the model say it again?* —
     and has no threshold or window to retune.
   - **Duplication is structurally impossible** (dup5 = 0 on testclip).
   - **A smaller step is worse on both axes.** Every commit resets the
     buffer, so small steps keep it short and reinstate the short-window
     failure mode: 0.3s → 20.2% WER, 0.2s → 27.9%, 0.1s → 45.2% — *and* small
     steps are what overflowed the inference budget.
   - **Agreement, not the pause heuristic, produces the accuracy.** With
     `--pause-secs 0` (every pass flushes — the old non-overlapping
     behaviour) WER is 102.9%.
   - **Remaining loss is vocabulary, not windowing.** On genesis-4, 20.3% of
     reference words are never decoded in any pass (`abel`, `adam`, `eve`,
     `seth`, `zillah`, …); only 2.3% are lost inside the streamer. See
     `backend/diagnose_loss.py`.
   - **Non-speech is skipped before the model sees it.** Whisper hallucinates
     stock phrases ("Thanks for watching") on music, and such a phantom never
     agrees, so it deadlocks the prefix match. A VAD gate plus a stall
     watchdog prevents that. See
     [`docs/streaming-latency.md`](docs/streaming-latency.md).
   - Measure inference with `backend/bench_pass.py` — it **consumes the lazy
     generator**, which timing the `transcribe()` call does not. Measuring at
     the call site reports ~3ms for work that takes ~500ms.
4. **VAD gate** — `faster-whisper` built-in Silero (`vad_filter=True`, `--vad-min-silence-ms=150`, `--vad-pad-ms=50`). There is deliberately **no** peak-amplitude gate: a fixed threshold reads quiet, compressed call audio as silence and drops real speech. Whichever is set is decided here, on full-bandwidth audio.
5. **Sentence assembler** — committed word texts accumulate in
   `SentenceAssembler` (pure, unit-tested). It receives only *committed* words,
   so there is no overlap to trim: the stable-prefix rule already decided what
   is new. Length-driven only: first line takes up to `--block-chars` (200)
   runes, every following line chains the last `--chain-chars` (15) runes plus
   fresh text; cuts land on word boundaries. A `draft` event per step carries
   the unstable tail; session stop flushes the remainder.
    - **Block size governs live-line churn, not commit latency.** The draft
      updates every `--step-secs` regardless, so a bigger block only makes
      text *permanent* less often.
    - **The live line is the unstable tail only.** Committed words are already
      in a final block, so including them in the draft would duplicate them.
    - `maxLineRunes` in the Go frontend (320) must stay above `--block-chars`,
      or every committed line and live draft truncates mid-sentence.
    - Cross-line repeats are **not** an artifact: each line chains the last
      `--chain-chars` (15) runes of the previous one by design, so a phrase
      spanning a commit boundary legitimately appears in both halves. Measure
      duplication *within* a line, or against the raw per-pass output.
6. **Inference** — `WhisperModel("small.en", device="cuda", compute_type="float16", download_root=MODEL_DIR)`, `language="en"`, `beam_size=5`, `word_timestamps=True`, `condition_on_previous_text=False`, `initial_prompt=None`.
   - `word_timestamps=True` is required by the stable-prefix rule and is free —
      measured within noise at every buffer length tested.
   - `medium.en` was measured and **rejected**: worse WER (51.6% vs 47.9%),
      2.8x slower, and it hallucinated conversational filler absent from the
      audio. Keep `small.en`.
   - Confidence gating, all passed to `transcribe()`:
      `no_speech_threshold=0.6`, `hallucination_silence_threshold=2.0`,
      `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`. These let
      Whisper's own certainty signal drop segments rather than displaying them
      as reliable — the documented fix for phantom phrases in silence, and the
      one that matters *more* on noisy audio, not less.
   - Measured (Quadro RTX 3000): ~0.003s per pass. Pass cost is linear in
      buffer length (0.027s at 12s), so the GPU is never the bottleneck.

### JSON protocol (stdout, one object per line, UTF-8, `ensure_ascii=False`)

```json
{"type": "status", "msg": "model loaded: small.en (cuda/float16)"}
{"type": "draft",  "text": "…in-progress line…", "t0": 12.5}
{"type": "final",  "text": "…120-char block…", "t0": 12.5, "t1": 18.5}
{"type": "error",  "msg": "CUDA DLLs not found…"}
```

No `partial` in v1 (locked decision). All logging → stderr. Run with `python -u` (unbuffered). The backend also watches stdin: a `quit` line triggers remainder-flush + clean exit (the Go UI uses this instead of killing).

### Flags

```
--model small.en --model-dir D:\earshot\models --device cuda --compute-type float16
--step-secs 0.3 --beam-size 5 --language en
```

Decoding gates: `--no-speech-threshold 0.6`, `--hallucination-silence-threshold 2.0`,
`--compression-ratio-threshold 2.4`, `--log-prob-threshold -1.0`.

## Go frontend (`go/`)

- `exec.Command("python", "-u", "backend/captions.py", …)`; goroutine scans stdout (`bufio.Scanner`, 1MB buffer) → `program.Send CaptionMsg`.
- Bubbles `viewport` + Lipgloss: **newest-first stack** of backend-assembled
  blocks, with the draft as a bright **live line + `▌` cursor** pinned on top
  while a sentence builds (chatbot feel). The newest committed line shares the
  live line's style, so the finished sentence and the one being generated read
  as one block — the cursor is what separates them — while older lines stay
  dimmed. 200-rune cap is a safety guard only (blocking/chaining lives in the
  backend). The transcript file keeps the full chronological finals — drafts
  never touch the file. Header records the exe build stamp + effective flags,
  so "am I running the new one?" is one glance.
- Keys: `space` pause/resume (UI-only: freeze display + stop appending), `q` / `ctrl+c` quit → kill subprocess → final save. `home` pins to newest, scrolling down browses history.
- Autosave: append each `final` to session file incrementally; on exit write header/footer. Path: `transcripts/YYYY-MM-DD_HHMM.md` (`--out-dir`).

## Paths — D: drive (locked)

C: has ~3GB free; D: has ~43GB free. All heavy artifacts live on D:.

| What | Canonical path | Override |
|---|---|---|
| Whisper models (`download_root`) | `D:\earshot\models` | `--model-dir` / `EARSHOT_MODEL_DIR` |
| HF hub cache | `D:\earshot\models\hf-hub` | `HF_HUB_CACHE` (set by `setup.bat`) |
| Python venv | `D:\earshot\.venv` (or project `.venv` if space allows) | — |
| Transcripts | `<project>/transcripts/` (tiny, fine on C:) | `--out-dir` (e.g. `D:\earshot\transcripts`) |
| CUDA DLLs | alongside venv or dir on `PATH` | `setup.bat` check |

`setup.bat` creates `D:\earshot\models`, sets `HF_HUB_CACHE`, verifies `nvidia-smi` + cuBLAS/cuDNN DLLs, then installs requirements. Never let `huggingface_hub` default back to `C:\Users\…\.cache`.

## Setup story

- `backend/requirements.txt` — pinned `faster-whisper`, `ctranslate2`, `PyAudioWPatch`, `numpy`, `scipy`.
- `setup.bat` — venv → pip install → CUDA check → device probe (`--list-devices`).
- Go: `go build -o earshot.exe ./go` — single `.exe` for the UI layer (Python runtime still required for backend in v1; PyInstaller deferred).

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA failed to load` at model init | Missing cuBLAS/cuDNN DLLs | Install `whisper-standalone-win` pack, add to `PATH` |
| Captions lag grows over time | Step too long for the machine | Lower `--step-secs` (costs accuracy — see the step-size table in docs/) |
| Duplicated phrases | Should be structurally impossible — only agreed words commit | Report it; the audio device may be shared with another earshot |
| Accuracy collapsed after an upgrade | A metric was self-referential (Whisper vs Whisper) | Score against `narrated-animations/videos/001_genesis-4/transcript/narration.srt` via `backend/score_stream.py` |
| First caption slow to appear | Model load, then the two-pass agreement dead zone | ~4s model load + ~2 steps (0.6s) |
| Repeated phantom phrases in silence | Hallucination on noise | Raise `--no-speech-threshold`, lower `--hallucination-silence-threshold`, keep `condition_on_previous_text=False` |
| Quiet speaker dropped entirely | Peak-amplitude gate misreading compressed audio | Fixed in v1.3 (VAD-only gating); if it returns, check for a custom `--silence-threshold` in your command line |
| No audio / wrong source | Default device changed | `python captions.py --list-devices`, pass `--device-index` |
| Cache re-downloads to C: | `HF_HUB_CACHE` unset | Re-run `setup.bat`; check env |
