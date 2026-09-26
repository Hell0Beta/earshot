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
3. **Window** — 0.5s sampler, 0.1s overlap (`--chunk-secs`, `--overlap-secs`). The live line updates ~2x/sec (AI-streaming feel); commits stay length-driven so small windows never spam lines.
4. **VAD gate** — `faster-whisper` built-in Silero (`vad_filter=True`, `--vad-min-silence-ms=150`, `--vad-pad-ms=50`; kept under the window length). Silent windows are skipped, never sent to the GPU.
5. **Sentence assembler** — chunk texts accumulate in `SentenceAssembler` (pure,
   unit-tested). Incoming chunks are overlap-trimmed against the buffer tail
   first (no boundary stutter). Length-driven only: first line takes up to
   `--block-chars` (200) runes, every following line chains the last
   `--chain-chars` (15) runes plus fresh text; cuts land on word boundaries.
   A `draft` event per chunk carries the growing line; session stop flushes
   the remainder. Every chunk also logs a timestamped heartbeat
   (state/peak/buffer) so a dark backend is always distinguishable from a
   silent room.
5. **Inference** — `WhisperModel("small.en", device="cuda", compute_type="float16", download_root=MODEL_DIR)`, `language="en"`, `beam_size=1, best_of=1` (greedy ≈ 2x faster than beam 5, fine for captions), `condition_on_previous_text=False` (chunks are independent — prevents hallucinated carry-over), `no_speech_threshold=0.6`.
   - Measured 2026-09-26 (Quadro RTX 3000): ~0.02s per chunk with speech
     present — far under even the 1s window. First chunk after idle
     pays a small GPU-wake tax; wall-clock chunking absorbs it.

### JSON protocol (stdout, one object per line, UTF-8, `ensure_ascii=False`)

```json
{"type": "status", "msg": "model loaded: small.en (cuda/float16)"}
{"type": "draft",  "text": "…in-progress line…", "t0": 12.5}
{"type": "final",  "text": "…200-char block…", "t0": 12.5, "t1": 18.5}
{"type": "error",  "msg": "CUDA DLLs not found…"}
```

No `partial` in v1 (locked decision). All logging → stderr. Run with `python -u` (unbuffered). The backend also watches stdin: a `quit` line triggers remainder-flush + clean exit (the Go UI uses this instead of killing).

### Flags

```
--model small.en --model-dir D:\earshot\models --device cuda --compute-type float16
--chunk-secs 6 --overlap-secs 0.5 --language en --list-devices
```

## Go frontend (`go/`)

- `exec.Command("python", "-u", "backend/captions.py", …)`; goroutine scans stdout (`bufio.Scanner`, 1MB buffer) → `program.Send CaptionMsg`.
- Bubbles `viewport` + Lipgloss: **newest-first stack** of backend-assembled
  blocks, with the draft as a bright **live line + `▌` cursor** pinned on top
  while a sentence builds (chatbot feel). 200-rune cap is a safety guard only
  (blocking/chaining lives in the backend). The transcript file keeps the full
  chronological finals — drafts never touch the file. Header records the exe
  build stamp + effective flags, so "am I running the new one?" is one glance.
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
| Captions lag grows over time | Chunk latency > window length | Drop to `base.en` or `int8_float16`, shorten `--chunk-secs` |
| Repeated phantom phrases in silence | Hallucination on noise | Raise `no_speech_threshold`, set `hallucination_silence_threshold`, keep `condition_on_previous_text=False` |
| No audio / wrong source | Default device changed | `python captions.py --list-devices`, pass `--device-index` |
| Cache re-downloads to C: | `HF_HUB_CACHE` unset | Re-run `setup.bat`; check env |
