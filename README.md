<p align="center">
  <img src=".github/assets/splash.png" alt="earshot" width="420" />
</p>

<h1 align="center">earshot</h1>

<p align="center">
  Live captions for whatever your speakers are playing &mdash; Teams, Zoom, a webinar, a video.
  Captured from system audio, transcribed on your own GPU, never leaves your machine.
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@hell0beta/earshot" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/npm/v/%40hell0beta%2Fearshot?style=for-the-badge&logo=npm&label=npm" alt="npm version" /></a>
  <a href="https://github.com/Hell0Beta/earshot" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Source-GitHub-blue?style=for-the-badge&logo=github" alt="View source on GitHub" /></a>
  <a href="LICENSE" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/github/license/Hell0Beta/earshot?style=for-the-badge" alt="License badge" /></a>
  <img src="https://img.shields.io/badge/platform-Windows%20only-0078D4?style=for-the-badge" alt="Windows only" />
  <img src="https://img.shields.io/badge/audio-100%25%20local-2E7D32?style=for-the-badge" alt="100% local" />
  <img src="https://img.shields.io/badge/cloud-none%20required-6A1B9A?style=for-the-badge" alt="No cloud" />
  <img src="https://img.shields.io/badge/GPU-optional%20(CPU%20fallback)-616161?style=for-the-badge" alt="GPU optional" />
</p>

---

## ✨ About

`earshot` is a terminal captioning tool for Windows. It records your audio output
via WASAPI loopback &mdash; so it hears *exactly* what you hear, from any app, with
no per-app integration &mdash; detects speech, and transcribes it on-device with
`faster-whisper`. Speech goes in, JSON lines come out, and a Bubble Tea TUI
renders them as a live caption that streams in word by word.

No audio is uploaded. No account, no API key, no network access at runtime. If you
pull the network cable it keeps working.

```sh
npx @hell0beta/earshot
```

<div align="center">
  <img src=".github/assets/hero.png" alt="earshot showing live captions" width="100%" />
</div>

---

## 🔥 Features

- 🎧 **Captures system audio, not your mic** &mdash; WASAPI loopback picks up
  Teams, Zoom, a browser tab, a game, anything routed to your output device
- 🔒 **100% local** &mdash; Whisper runs on your own GPU via CUDA; no audio ever
  leaves the machine
- ⚡ **Live line, not a slideshow** &mdash; the in-progress sentence grows word by
  word as each window is transcribed, then commits into 120-character blocks.
  The window is 1.5&nbsp;s, so you read roughly 0.75&nbsp;s behind the speaker.
- 🧩 **Sentence blocks, newest first** &mdash; readable chunks with a blank line
  separating the newest from the history, wrapped to your terminal width
- 🎨 **Animated 5&times;5 splash** &mdash; a pixel-art `EARSHOT` banner with a
  travelling highlight, plus a mini banner in the header while paused
- ⏸️ **Pause without losing audio** &mdash; freeze the display, the backend keeps
  listening; unpause and the sentence picks up where it left off
- 📝 **Autosaved transcripts** &mdash; every session written to a timestamped
  `.md`, with an automatic retention cap (30 sessions by default)
- 🩺 **`earshot doctor`** &mdash; one command that tells you exactly what is
  misconfigured and how to fix it
- 🖥️ **GPU optional** &mdash; no CUDA? It falls back to CPU automatically and
  tells you so
- 🪶 **Single 5 MB binary** &mdash; no runtime for the UI layer; the Python
  backend is bootstrapped once by `earshot setup`

---

## 🚀 Quick Start

### One-off (no install)

```powershell
npx @hell0beta/earshot setup   # first time only: venv + backend deps
npx @hell0beta/earshot         # start captioning
```

### Global install

```powershell
npm install -g @hell0beta/earshot

earshot setup
earshot
```

### From source

```powershell
git clone https://github.com/Hell0Beta/earshot.git
cd earshot
npm run prepack        # builds vendor/earshot.exe
node bin/earshot.js setup
node bin/earshot.js
```

Requirements: **Windows 10/11 x64**, Python 3.9+ on `PATH` (only for the one-time
`setup`), and Node 14+ if you're using the npm wrapper.

---

## ⌨️ Keys

| Key            | Action                                              |
| -------------- | --------------------------------------------------- |
| `space`        | pause / resume the display (backend keeps listening) |
| `q` / `ctrl+c` | quit &mdash; flushes the last sentence and saves    |
| `home`         | jump to the newest caption                          |
| `end`          | jump to the oldest                                  |
| `pgup`/`pgdn`  | scroll the history                                  |
| `↑` / `↓`      | scroll line by line                                  |

> **Quit with `q`, not the window's X.** Closing the window kills the backend
> mid-sentence and the tail of your transcript is lost.

---

## ⚙️ Options

| Flag                | Description                                              | Default              |
| ------------------- | -------------------------------------------------------- | -------------------- |
| `--model NAME`      | Whisper model (`small.en`, `base.en`, `medium.en`, `turbo`) | `small.en`         |
| `--device WHAT`     | `auto`, `cuda` or `cpu`                                  | `auto`               |
| `--compute-type T`  | CTranslate2 compute type                                 | auto per device      |
| `--chunk-secs N`    | capture window length (also the worst-case caption lag)   | `1.5`                |
| `--overlap-secs N`  | window overlap                                           | `0.2`                |
| `--block-chars N`   | characters per committed sentence                        | `120`                |
| `--chain-chars N`   | trailing characters chained into the next block          | `15`                 |
| `--out-dir DIR`     | where transcripts are written                            | `<data>/transcripts` |
| `--keep N`          | transcript sessions to retain (`0` keeps none)           | `30`                 |
| `--no-prune`        | never delete old transcripts                             | off                  |
| `--splash-style S`  | `shimmer` or `rainbow`                                   | `shimmer`            |
| `--python PATH`     | interpreter to use                                       | auto                 |
| `--backend PATH`    | backend script                                           | auto                 |
| `--model-dir DIR`   | model cache                                              | `<data>/models`      |
| `--cuda-dll-dir D`  | folder holding cuBLAS/cuDNN, prepended to `PATH`         | auto                 |

Environment: `EARSHOT_HOME` (data dir, default `%LOCALAPPDATA%\earshot`),
`EARSHOT_PYTHON`, `EARSHOT_CUDA_DLL_DIR`.

---

## 🔧 Setup, GPU and CPU

```sh
earshot setup     # create the Python venv, install backend deps
earshot doctor    # verify everything resolves
earshot           # start
```

**CPU-only machines work out of the box.** Without CUDA, earshot switches to
`int8` on the CPU and prints a note. That is slower than real-time for
`small.en`; use `--model base.en` there.

**For GPU acceleration** (optional, and the fast path) install the
[cuBLAS/cuDNN for CUDA 12](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs)
pack, then either add it to your `PATH` or point earshot at it:

```powershell
$env:EARSHOT_CUDA_DLL_DIR = "D:\cuBLAS.and.cuDNN_CUDA12_win_v3"
earshot doctor    # "GPU visible to ctranslate2: 1 CUDA device(s)"
```

The first run downloads the model into `<data>/models` (~460 MB for `small.en`).
The animated splash covers the wait.

### Where files live

Models, transcripts and the fallback venv share one writable root, called the
**data dir**:

| Platform | Default                                                  |
| -------- | -------------------------------------------------------- |
| Windows  | `%LOCALAPPDATA%\earshot`                                  |
| Other    | `~/.earshot`                                              |

Set `EARSHOT_HOME` to move it &mdash; useful when `%LOCALAPPDATA%` is small, or
to keep a large model cache on a faster or roomier drive:

```powershell
[Environment]::SetEnvironmentVariable("EARSHOT_HOME", "D:\earshot", "User")
```

Run `earshot doctor` to confirm. If the variable is unset, `earshot doctor`
prints a `[warn]` line naming the folder that will be used, because a silent
~460 MB download on first run surprises people who already have the model
somewhere else.

> A shell that was already open when you set the variable keeps its old
> environment and will still resolve the default. Open a new terminal, or set
> `$env:EARSHOT_HOME` in the current one.

### Choosing a window length and a model

Measured on a Quadro RTX 3000, scoring word precision against known speech
(higher is better, `stutters` counts repeated word pairs):

| Model | Window | Precision | Stutters |
| --------- | ------ | --------- | -------- |
| `small.en` | 1.5s  | **0.93**  | 0        |
| `medium.en`| 1.5s  | 0.91      | 0        |
| `small.en` | 1.0s  | 0.86      | 0        |
| `medium.en`| 1.0s  | 0.79      | 0        |

Two things worth knowing, because both are counter-intuitive:

- **Window length matters more than model size.** A bigger model on a short
  window is *worse*, not better &mdash; `medium.en` invents whole phrases
  (`description`, `devices`, `video`) on half a second of audio, where
  `small.en` on 1.5s does not.
- **Precision, not recall, is the thing to watch.** Every config captured
  100% of the reference words. Bad output is *extra* invented words, never
  missing ones &mdash; so if captions look wrong, make the window longer
  rather than reaching for a larger model.

On CPU, prefer `base.en`; `small.en` is slower than real time there.

---

## 🧠 How it works

| Path                     | Purpose                                                                    |
| ------------------------ | -------------------------------------------------------------------------- |
| `backend/captions.py`    | WASAPI loopback &rarr; 16 kHz mono &rarr; Silero VAD &rarr; Whisper &rarr; JSON lines |
| `backend/test_assembler.py` | unit tests for the sentence assembler (no audio or GPU needed)          |
| `go/main.go`             | TUI: spawns the backend, parses events, renders captions, saves transcripts |
| `go/paths.go`            | path resolution so it runs from a checkout, a shim or `node_modules`        |
| `go/doctor.go`           | `setup` and `doctor` subcommands                                            |
| `go/splash.go`           | the pixel banner and its animations                                         |
| `go/splash_art.go`       | the banner, generated once with [`bit`](https://github.com/superstarryeyes/bit) and baked in |
| `go/transcripts.go`      | transcript retention                                                        |
| `bin/earshot.js`         | npm launcher; hands over to the native binary                              |
| `ARCHITECTURE.md`        | design notes and the decisions behind them                                  |

### The pipeline

```
speakers ──▶ WASAPI loopback ──▶ 16 kHz mono ──▶ VAD gate ──▶ Whisper (CUDA)
                                                                    │
                     transcript file ◀── TUI ◀── JSON lines ◀───────┘
```

The backend captures overlapping windows, drops the silent ones before they ever
reach the GPU, and assembles the surviving text into fixed-size blocks
character-driven rather than time-driven. Only complete blocks are committed; the
partially-built sentence is streamed to the TUI as a live line.

Protocol, if you want to build on it: one JSON object per line on stdout
&mdash; `status`, `draft`, `final`, `error`. All logging goes to stderr, and a
`.backend.log` is written next to each transcript.

---

## 🐛 Troubleshooting

| Symptom                                     | Cause and fix                                                                     |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| Nothing appears, footer says `starting backend…` | The backend died. Read the `.backend.log` next to the transcript &mdash; it has the exact error. |
| Nothing appears, says `○ backend down`     | Backend exited. `earshot doctor` will say which check failed.                        |
| No captions but audio is playing           | Wrong capture device. `earshot doctor` &rarr; "audio capture device"; plug in headphones or pass `--device-index`. |
| Captions lag behind the audio              | Lower `--chunk-secs` (e.g. `1.0`). Accuracy drops with it &mdash; 1.5s measured 0.93 precision vs 0.86 at 1.0s. |
| Repeated or garbled words                  | Overlap too small for the model. Try `--overlap-secs 0.3`.  |
| Whispers up extra words that were never said | Short windows make it confabulate. **Raise** `--chunk-secs`, do not lower it. |
| Phantom phrases during silence             | Whisper hallucinating on noise. `--block-chars 200` and a larger model help.       |
| `CUDA failed with error no CUDA-capable device` | GPU is asleep or held. See the note below.                                  |
| Another earshot is already running         | **Only run one at a time** &mdash; two backends on one audio device will fight, and the second can wedge. |
| The GPU vanished after a restart (Code 45) | A laptop power-gating quirk, not earshot. Cold boot on AC: `shutdown /s /t 0`, hold power 15s, boot plugged in. |
| First inference seems to hang for a few seconds | Normal: the first CUDA call loads and autotunes kernels. Windows' 2-second GPU watchdog used to kill it; raise it with `TdrDelay=30` if you hit `LiveKernelEvent 141` in the event log. |
| First run took minutes and downloaded the model | The model downloads on first run (~460 MB for `small.en`, ~1.5 GB for `medium.en`). Later runs start in ~2s. If you keep models on another drive, set `EARSHOT_HOME` so the cache is not re-fetched. |
| Transcripts filling the disk               | `earshot --keep 10`, or `--no-prune` to disable entirely.                           |

`earshot doctor` is the first thing to run for any of the above.

---

## 📄 License

[MIT](LICENSE) &copy; 2026 Hell0Beta

Bundles and builds on:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) and
  [CTranslate2](https://github.com/OpenNMT/CTranslate2) (MIT)
- [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) (MIT)
- [bit](https://github.com/superstarryeyes/bit) (MIT) &mdash; the splash banner is
  rendered with its `fivebyfive` font and baked in at build time
- OpenAI Whisper model weights (MIT), downloaded on first run
