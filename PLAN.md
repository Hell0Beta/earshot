# Earshot — Live Local Subtitles for System Audio

## Problem

On Windows there is no easy way to get live, private, local subtitles of what is
coming through your speakers — Teams, Zoom, a webinar, whatever. Existing options
either only capture your mic, require sending audio to the cloud, or are locked
inside a specific app's UI. Earshot is a terminal tool you launch once that shows
live captions of your system audio, fully offline, using the GPU you already have.

## Solution

Two-part local application:

- **Python backend** — captures PC speaker output (WASAPI loopback), gates on
  speech (VAD), transcribes with `faster-whisper` on CUDA. Emits transcript
  events as structured JSON lines on stdout.
- **Go frontend** — a Bubble Tea terminal UI that launches the Python backend as
  a subprocess, reads its JSON output, and renders a clean live subtitle view.
  Handles pause/quit and saves the transcript on exit.

Python does the audio/ML heavy lifting (where the best tooling exists); Go owns
the process and the interface (fast, good-looking terminal, single `.exe` for
the UI layer).

## Locked decisions (2026-09-26)

| Decision | Choice | Why |
|---|---|---|
| Default model | `small.en` | Best speed/accuracy balance on 6GB Quadro RTX 3000 (~2GB in float16, headroom left) |
| Caption granularity | 0.5s sampler, 200-char sentence blocks | Live line grows ~2×/sec like AI streaming; commits on length only (first 200, then tail-15 + fresh); remainder flushes on stop |
| Caption display | Bright live line + cursor over newest-first stack | In-progress sentence prominent on top; solid 200-char blocks below; transcript file keeps full chronological finals |
| Pause semantics | UI-only pause | Go freezes display + stops saving; Python keeps running. No stdin control protocol, no audio-gap edge cases |
| Model location | **`D:\earshot\models`** (canonical) | C: has ~3GB free; models + CUDA must live on D: (43GB free). Override via `--model-dir` / `EARSHOT_MODEL_DIR` |
| Language | `en` fixed for v1 | `.en` models are smaller/faster; multilingual later |
| VAD | Built-in Silero via `vad_filter=True` | One less dependency vs standalone webrtcvad; split out only if chunks lag |

## Deliverables

1. **Python backend** — loopback capture → resample → VAD → GPU Whisper → JSON lines (`final` / `status` / `error`)
2. **Go CLI with Bubble Tea UI** — spawns backend, parses events, live captions view (large readable text, minimal chrome, auto-scroll), pause/resume/quit
3. **Transcript autosave** — incremental append during session + final file `transcripts/YYYY-MM-DD_HHMM.md` on exit (crash-safe, not exit-only)
4. **Config/flags** — model size, `--model-dir`, compute type, chunk length, output folder, `--list-devices`
5. **Setup story** — `requirements.txt` + venv + `setup.bat` (CUDA DLL check included), Go build → single `.exe`

## Milestones

1. **M1 — Audio probe.** `backend` lists WASAPI devices, opens default loopback, dumps 5s WAV. Proves capture + resample path before any ML.
2. **M2 — Transcription loop.** `small.en` on CUDA/float16, VAD-gated 6s windows, JSON-lines to stdout. Target: chunk latency < window length (no growing lag).
3. **M3 — TUI.** Go spawns backend, viewport render, space=pause, q=quit+save, incremental autosave.
4. **M4 — Hardening.** CUDA DLL check, `--list-devices`, hallucination tuning, pinned requirements, docs.

## Risks

1. **CUDA on Windows.** `ctranslate2` needs CUDA 12 + cuDNN 9 DLLs on `PATH` (Purfview `whisper-standalone-win` pack). `setup.bat` must verify before model load.
2. **Resampling.** Loopback is 44.1/48kHz stereo; Whisper needs 16kHz mono. Explicit downmix + resample required.
3. **Stdout discipline.** `python -u`, logs → stderr, stdout = JSON only, UTF-8. Go scanner needs enlarged buffer.
4. **Disk.** C: is nearly full — everything heavy (models, HF hub cache, CUDA wheels) goes to D:. See `ARCHITECTURE.md` § Paths.

## Non-goals (v1)

Diarization, word-level streaming, per-app capture, cloud fallback, PyInstaller bundle (~2GB with CUDA DLLs — venv first).
