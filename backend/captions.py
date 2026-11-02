"""Earshot backend: WASAPI loopback -> VAD -> faster-whisper -> JSON lines.

Modes:
  --list-devices   show capture devices (M1)
  --live           continuous transcription loop (M2), emits final/status/error
  (default)        short WAV dump for probe/verification (M1)

Protocol discipline:
  - stdout = JSON lines only (status / final / error)
  - everything human-readable -> stderr
  - run with python -u (unbuffered)
"""

import argparse
import collections
import csv
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
import wave

import numpy as np
from scipy.signal import resample_poly  # eager: pay the ~1.5s import at startup,
                                        # never silently mid-stream (M3 stall)

TARGET_RATE = 16000
DEFAULT_MODEL_DIR = os.environ.get("EARSHOT_MODEL_DIR", r"D:\earshot\models")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def parse_command(line):
    """Interpret one stdin line from the TUI.

    Returns (kind, payload): ("quit", None), ("polish", text), or (None, None)
    for anything unrecognised. Kept separate from the reader thread so the
    protocol can be tested without a microphone or a model.
    """
    line = line.strip()
    if not line:
        return None, None
    if line.lower() == "quit":
        return "quit", None
    try:
        cmd = json.loads(line)
    except ValueError:
        return None, None
    if not isinstance(cmd, dict):
        return None, None
    if cmd.get("cmd") == "polish":
        return "polish", str(cmd.get("text") or "")
    if cmd.get("cmd") == "source" and cmd.get("kind") in ("mic", "loopback"):
        return "source", cmd["kind"]
    if cmd.get("cmd") == "flush":
        return "flush", None
    if cmd.get("cmd") == "reset":
        return "reset", None
    return None, None


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def find_loopback(p, device_index=None):
    import pyaudiowpatch as pyaudio

    if device_index is not None:
        info = p.get_device_info_by_index(device_index)
        log("Using device #%d: %s" % (device_index, info["name"]))
        return info
    try:
        loopback = p.get_default_wasapi_loopback()
        log("Using default WASAPI loopback: %s" % loopback["name"])
        return loopback
    except Exception:
        pass
    # Fallback: match default speakers to a loopback analogue by name.
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        raise RuntimeError("WASAPI is not available on this system")
    speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    for loopback in p.get_loopback_device_info_generator():
        if speakers["name"] in loopback["name"]:
            log("Matched loopback: %s" % loopback["name"])
            return loopback
    raise RuntimeError(
        "Default loopback output device not found. "
        "Run with --list-devices to pick --device-index."
    )


def find_mic(p, device_index=None):
    """The real microphone, never a WASAPI loopback device.

    pyaudiowpatch lists loopback endpoints as input devices too, so an
    "input=True" stream on the default input can still be the speakers on
    some systems. Dictation must never capture what the machine is playing.
    """
    if device_index is not None:
        info = p.get_device_info_by_index(device_index)
        if info.get("isLoopbackDevice"):
            raise RuntimeError(
                "device #%d (%s) is a loopback of the speakers, not a "
                "microphone" % (device_index, info["name"]))
        if int(info["maxInputChannels"]) < 1:
            raise RuntimeError("device #%d (%s) has no input channels"
                               % (device_index, info["name"]))
        log("Using input device #%d: %s" % (device_index, info["name"]))
        return info
    import pyaudiowpatch as pyaudio
    # WASAPI first: the legacy MME default input is the same microphone with
    # ~100ms more buffering, and MME truncates long endpoint names.
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        info = p.get_device_info_by_index(wasapi["defaultInputDevice"])
        if (not info.get("isLoopbackDevice")
                and int(info["maxInputChannels"]) >= 1):
            log("Using default WASAPI microphone: %s" % info["name"])
            return info
    except OSError:
        pass
    try:
        info = p.get_default_input_device_info()
        if not info.get("isLoopbackDevice"):
            log("Using default microphone: %s" % info["name"])
            return info
    except OSError:
        pass
    for info in p.get_device_info_generator():
        if (int(info["maxInputChannels"]) >= 1
                and not info.get("isLoopbackDevice")):
            log("Using microphone: %s" % info["name"])
            return info
    raise RuntimeError(
        "No microphone found. Run with --list-devices to pick --device-index.")


def find_source(p, kind, device_index=None):
    return find_mic(p, device_index) if kind == "mic" \
        else find_loopback(p, device_index)


def resample_to_16k(mono, src_rate):
    """Mono float32 at src_rate -> float32 at 16 kHz."""
    if src_rate == TARGET_RATE:
        return mono.astype(np.float32)
    g = math.gcd(TARGET_RATE, src_rate)
    up, down = TARGET_RATE // g, src_rate // g
    return resample_poly(mono, up, down).astype(np.float32)


def bytes_to_mono16k(raw, channels, src_rate):
    # Truncate to whole int16 samples before frombuffer. numpy raises
    # "buffer size must be a multiple of element size" on an odd byte count,
    # which killed the backend whenever a step produced one.
    raw = bytes(raw)
    if len(raw) % 2:
        raw = raw[:-1]
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio[: (len(audio) // channels) * channels].reshape(-1, channels)
        audio = audio.mean(axis=1)
    return resample_to_16k(audio, src_rate)


def frame_bytes_for(channels):
    """Bytes per audio frame in the device's own format.

    The step size must be snapped to this: a byte count that is not a whole
    number of frames cannot be reshaped to (-1, channels), and int16 needs
    an even length regardless.
    """
    return max(1, int(channels) * 2)


def cmd_list_devices(p):
    devs = []
    for info in p.get_device_info_generator():
        devs.append(
            {
                "index": info["index"],
                "name": info["name"],
                "inputs": info["maxInputChannels"],
                "outputs": info["maxOutputChannels"],
                "rate": info["defaultSampleRate"],
                "loopback": bool(info.get("isLoopbackDevice")),
            }
        )
        log(
            "#%(index)d %(name)s | in=%(inputs)d out=%(outputs)d "
            "%(rate)sHz%(lb)s"
            % {
                "index": info["index"],
                "name": info["name"],
                "inputs": info["maxInputChannels"],
                "outputs": info["maxOutputChannels"],
                "rate": info["defaultSampleRate"],
                "lb": " [LOOPBACK]" if info.get("isLoopbackDevice") else "",
            }
        )
    emit({"type": "status", "msg": "devices listed", "devices": devs})


def open_input_stream(p, device, frames, q):
    import pyaudiowpatch as pyaudio

    def callback(in_data, frame_count, time_info, status):
        q.put(bytes(in_data))
        return (in_data, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paInt16,
        channels=int(device["maxInputChannels"]),
        rate=int(device["defaultSampleRate"]),
        frames_per_buffer=frames,
        input=True,
        input_device_index=device["index"],
        stream_callback=callback,
    )
    stream.start_stream()
    return stream


def cmd_dump(p, args):
    device = find_loopback(p, args.device_index)
    channels = int(device["maxInputChannels"])
    rate = int(device["defaultSampleRate"])
    frames = int(args.frames_per_buffer)

    q = queue.Queue()
    emit(
        {
            "type": "status",
            "msg": "capturing",
            "device": device["name"],
            "src_rate": rate,
            "channels": channels,
            "seconds": args.dump_secs,
        }
    )
    log(
        "Capturing %.1fs from '%s' (%dHz, %dch) — play some audio now…"
        % (args.dump_secs, device["name"], rate, channels)
    )
    stream = open_input_stream(p, device, frames, q)
    log("stream started, collecting…")
    try:
        n_chunks = int(rate * args.dump_secs / frames) + 1
        chunks = []
        for i in range(n_chunks):
            try:
                chunks.append(q.get(timeout=8))
            except queue.Empty:
                raise RuntimeError(
                    "no audio data flowed from the loopback device "
                    "(is anything playing? else: exclusive-mode holder?)"
                )
        raw = b"".join(chunks)
    finally:
        stream.stop_stream()
        stream.close()

    mono = bytes_to_mono16k(raw, channels, rate)
    peak = float(np.abs(mono).max()) if len(mono) else 0.0

    with wave.open(args.dump_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes((np.clip(mono, -1, 1) * 32767).astype(np.int16).tobytes())

    emit(
        {
            "type": "status",
            "msg": "dump written",
            "file": args.dump_file,
            "peak": round(peak, 4),
            "silent": peak < 0.001,
        }
    )
    log("Wrote %s (peak=%.4f%s)" % (args.dump_file, peak, " — SILENT!" if peak < 0.001 else ""))


def load_model(args):
    from faster_whisper import WhisperModel

    os.makedirs(args.model_dir, exist_ok=True)
    log("Loading model '%s' (%s/%s) from %s …"
        % (args.model, args.device, args.compute_type, args.model_dir))
    t0 = time.time()
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=args.model_dir,
    )
    log("Model ready in %.1fs" % (time.time() - t0))
    emit(
        {
            "type": "status",
            "msg": "model loaded",
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
        }
    )
    return model


class SentenceAssembler:
    """Accumulate chunk texts into fixed-size display blocks.

    Length-driven, never time-driven: the first line takes up to
    `block_chars` runes; every following line chains the last `chain_chars`
    runes of the previous line plus fresh text up to
    `block_chars - chain_chars - 1` (the -1 is the joining space), so every
    emitted line stays within `block_chars`. Cuts land on word boundaries
    (hard cut only for a spaceless run longer than the limit); the remainder
    stays buffered. `flush()` emits whatever is left (e.g. on session stop).
    buf_t0 is the start of the oldest unconsumed text.

    Receives only *committed* words from the streamer, so there is nothing to
    dedup here: RollingStreamer's stable-prefix rule already decided what is
    new. The earlier design trimmed overlap by text similarity and added error
    faster than it removed it (167.5% WER against a Genesis 4 reference); see
    docs/streaming-latency.md for the full record.
    """

    def __init__(self, block_chars=200, chain_chars=15):
        self.block_chars = block_chars
        self.chain_chars = chain_chars
        self.buf = ""
        self.buf_t0 = 0.0
        self.buf_t1 = 0.0
        self.prev_tail = ""
        self.first = True

    def reset(self):
        """Forget pending text (the user cleared the transcript)."""
        self.buf = ""
        self.buf_t0 = 0.0
        self.buf_t1 = 0.0
        self.prev_tail = ""
        self.first = True

    @staticmethod
    def _take(text, limit):
        if len(text) <= limit:
            return text, ""
        cut = text.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        return text[:cut], text[cut:].lstrip()

    @staticmethod
    def _norm(word):
        """Canonical form for word comparisons (exported for tests)."""
        return word.strip().strip("\"'“”‘’()[]{}.,.!?;:…-_").lower()

    def _compose(self, core):
        if self.first or not self.prev_tail:
            return core
        return (self.prev_tail + " " + core).strip()

    def _emit_head(self, head):
        line = self._compose(head)
        out = (line, self.buf_t0, self.buf_t1)
        self.prev_tail = line[-self.chain_chars:] if len(line) > self.chain_chars else line
        self.first = False
        # The remainder left after the cut belongs to the text that follows
        # this block, so its clock starts where the block ended. Without this
        # the buffer's start time never advances and every later block
        # inherits a stale (identical) timestamp.
        self.buf_t0 = out[2]
        return out

    def add_chunk(self, text, t0, t1):
        """Feed one chunk's text; returns a list of (line, t0, t1) finals."""
        text = text.strip()
        if not text:
            return []
        if self.buf:
            self.buf += " " + text
        else:
            self.buf = text
            self.buf_t0 = t0
        self.buf_t1 = t1
        finals = []
        while True:
            if self.first:
                if len(self.buf) < self.block_chars:
                    break
                head, self.buf = self._take(self.buf, self.block_chars)
            else:
                target = self.block_chars - self.chain_chars - 1
                if len(self.buf) < target:
                    break
                head, self.buf = self._take(self.buf, target)
            finals.append(self._emit_head(head))
        return finals

    def draft(self):
        """Current in-progress line composition (display-only)."""
        if not self.buf.strip():
            return "", 0.0
        return self._compose(self.buf.strip()), self.buf_t0

    def flush(self):
        """Emit the remainder, however short. Returns (line, t0, t1) or None.

        Also ends the chain: an explicit flush is a boundary the user asked
        for (they are about to edit or copy), so the next live line must not
        start with the tail of the line just committed.
        """
        if not self.buf.strip():
            return None
        self.buf = self.buf.strip()
        out = self._emit_head(self.buf)
        self.buf = ""
        self.prev_tail = ""
        self.first = True
        return out


def compose_live_line(assembler, unstable):
    """Build the live (`draft`) line: pending block + unstable tail.

    The block is every committed word in the current, not-yet-final line --
    that is what makes the sentence grow in front of the user instead of
    appearing a word at a time. The unstable tail (words the streamer has
    heard but not yet had confirmed twice) is appended so the line ends
    where the audio currently is, and those last words visibly settle as
    the next passes agree with each other.

    Showing committed text here is not duplication: `draft` is its own
    display row, and the same text is carried into the `final` event when
    the block fills. The live line is meant to preview that block.

    Returns (text, t0).
    """
    base, t0 = assembler.draft()
    tail = " ".join(w.text for w in unstable).strip()
    if base and tail:
        return base + " " + tail, t0
    if tail:
        # No committed block yet, so the tail carries its own start time.
        return tail, unstable[0].start
    return base, t0

def plan_step(floor_secs, headroom, pass_hist, max_step_secs=None):
    """How much audio the next cycle may consume.

    Sized from the *measured* cost of recent passes, not from a fixed number.
    One pass over ~0.5s of real speech costs far more than a pass over 0.3s
    (measured: 111 of 131 real passes exceeded 100ms, p90 ~590ms), so a fixed
    small step cannot keep up -- every cycle overruns the audio it represents,
    the queue grows, and caption lag increases without bound. That is the
    accumulating-latency bug.

    The estimator is the **median**, not the p90. p90 was chosen to avoid a
    slow spike inflating every later step, but it does the opposite when the
    history is short: one 18s pass (GPU contention) sits above the 90th
    percentile of a 12-sample window, so `p90 * headroom` jumped to 16.8s and
    then 19.5s. The step then waited that long before decoding, which froze the
    captions and produced a 44-word burst when the buffer cap finally flushed.
    A median is unmoved by one outlier.

    `max_step_secs` is a hard ceiling so no estimator can ever recreate that
    runaway; a step larger than the buffer is pointless anyway.
    """
    if not pass_hist:
        step = floor_secs
    else:
        ordered = sorted(pass_hist)
        mid = len(ordered) // 2
        med = (ordered[mid] if len(ordered) % 2 else
               0.5 * (ordered[mid - 1] + ordered[mid]))
        step = max(floor_secs, med * headroom)
    if max_step_secs is not None:
        step = min(step, max_step_secs)
    return step


class LatencyLog:
    """Per-word spoken -> committed latency, written as CSV.

    The backend log is prose for humans; this is the machine-readable artifact
    you can plot. One row per committed word, plus event rows for cycles that
    committed nothing (gate skips, watchdogs, resyncs) so a gap in the data is
    explainable rather than invisible.

    Columns: wall, cycle, event, word, audio_end_s, latency_s, pass_s,
    target_s, buffer_s.
    """

    HEADER = ["wall", "cycle", "event", "word", "audio_end_s",
              "latency_s", "pass_s", "target_s", "buffer_s"]

    def __init__(self, path):
        self.path = path
        self.f = None
        self.latencies = []
        self.words = 0
        self.events = collections.Counter()
        if not path:
            return
        try:
            self.f = open(path, "w", encoding="utf-8", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(self.HEADER)
            self.f.flush()
        except OSError as exc:
            log("latency log disabled: %s" % exc)
            self.f = None

    def _row(self, cycle, event, word, audio_end, latency, pass_s, target, buf):
        if not self.f:
            return
        self.w.writerow(["%.3f" % time.time(), cycle, event, word,
                         "%.2f" % audio_end, "%.3f" % latency,
                         "%.3f" % pass_s, "%.3f" % target, "%.2f" % buf])
        self.f.flush()

    def add_word(self, cycle, word, latency, pass_s, target, buf):
        self.latencies.append(latency)
        self.words += 1
        self._row(cycle, "word", word.text, word.end, latency,
                  pass_s, target, buf)

    def add_event(self, cycle, event, pass_s, target, buf):
        self.events[event] += 1
        self._row(cycle, event, "", 0.0, 0.0, pass_s, target, buf)

    def summary(self):
        """One-line distribution for the backend log."""
        if not self.latencies:
            ev = ", ".join("%s=%d" % kv for kv in sorted(self.events.items()))
            return "latency: no committed words" + (" (%s)" % ev if ev else "")
        ordered = sorted(self.latencies)
        p50 = ordered[len(ordered) // 2]
        p90 = ordered[min(len(ordered) - 1, int(round(0.9 * len(ordered))) - 1)]
        return ("latency: %d words, median %.2fs, p90 %.2fs, max %.2fs"
                % (self.words, p50, p90, ordered[-1]))

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


class AudioWriter:
    """Append the captured mono 16 kHz audio to a WAV as the session runs.

    Without this a live session leaves no audio, so a reported freeze can only
    be argued about, never replayed. With it, any stall in `.backend.log` maps
    to a timestamp in the WAV and can be re-decoded offline -- which is exactly
    how the temperature-fallback stalls were found.

    Digital silence is written for the zero-padding `collect_step` adds when the
    device is idle, so the file stays aligned with session time.
    """

    def __init__(self, path):
        self.w = None
        if not path:
            return
        try:
            self.w = wave.open(path, "wb")
            self.w.setnchannels(1)
            self.w.setsampwidth(2)
            self.w.setframerate(TARGET_RATE)
        except OSError as exc:
            log("audio dump disabled: %s" % exc)
            self.w = None

    def add(self, mono):
        if self.w is None:
            return
        x = np.clip(mono, -1.0, 1.0)
        self.w.writeframes((x * 32767.0).astype("<i2").tobytes())

    def close(self):
        if self.w is not None:
            # wave rewrites the header with the true frame count on close.
            self.w.close()
            self.w = None


def cmd_live(p, args):
    source_kind = args.input
    try:
        device = find_source(p, source_kind, args.device_index)
    except Exception as exc:
        # A v2t session without a usable microphone should still start: the
        # user can fix the device and switch again, and live captions keep
        # working in the meantime.
        if source_kind != "mic":
            raise
        emit({"type": "status",
              "msg": "no microphone (%s) — using system audio" % exc})
        source_kind = "loopback"
        device = find_loopback(p, args.device_index)
    channels = int(device["maxInputChannels"])
    rate = int(device["defaultSampleRate"])
    frames = int(args.frames_per_buffer)
    bytes_per_sec = rate * channels * 2  # int16

    emit({"type": "status", "msg": "loading model (one-time download on first run)…"})
    model = load_model(args)

    q = queue.Queue()
    stream = open_input_stream(p, device, frames, q)

    step_secs = args.step_secs
    assembler = SentenceAssembler(args.block_chars, args.chain_chars)

    from streaming import RollingStreamer

    # `full` leaves faster-whisper's default temperature list untouched (None
    # means "do not pass the argument"). The others are the measured policies
    # from the freeze investigation: `off` removed 43 of 44 slow passes.
    temperature = {"full": None,
                   "short": [0.0, 0.2, 0.4],
                   "off": 0.0}[args.temp_fallback]

    streamer = RollingStreamer(
        model,
        language=args.language,
        max_buf_secs=args.max_buf_secs,
        pause_secs=args.pause_secs,
        beam_size=args.beam_size,
        prompt_chars=args.prompt_chars,
        min_speech_secs=args.min_speech_secs,
        gate_min_buf_secs=args.gate_min_buf_secs,
        stall_passes=args.stall_passes,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_pad_ms=args.vad_pad_ms,
        no_speech_threshold=args.no_speech_threshold,
        hallucination_silence_threshold=args.hallucination_silence_threshold,
        compression_ratio_threshold=args.compression_ratio_threshold,
        log_prob_threshold=args.log_prob_threshold,
        temperature=temperature,
    )

    emit(
        {
            "type": "status",
            "msg": "live",
            "device": device["name"],
            "source": source_kind,
            "step_secs": step_secs,
            "block_chars": args.block_chars,
            "beam_size": args.beam_size,
        }
    )
    log("Live transcription started — Ctrl+C to stop.")

    stop_event = threading.Event()
    # The TUI says which source the active mode needs; the audio loop picks the
    # change up between cycles rather than reopening the stream from a thread.
    source_lock = threading.Lock()
    want_source = {"kind": source_kind}
    # `flush` asks for the pending words to be committed *now*, so the TUI can
    # open its editor with the complete document instead of dropping the
    # sentence that has not agreed with a second hypothesis yet.
    flush_req = threading.Event()
    # `reset` is the TUI's "clear the transcript": forget the pending words so
    # nothing the user cleared can resurface in a later update.
    reset_req = threading.Event()

    def watch_stdin():
        """Commands from the TUI: `quit`, a polish request, or a source switch.

        The TUI owns the document text (it has the corrections and the user's
        edits), so it sends the text to polish rather than asking the backend to
        reconstruct it.
        """
        try:
            for line in sys.stdin:
                kind, payload = parse_command(line)
                if kind == "quit":
                    stop_event.set()
                    break
                if kind == "polish" and corrector is not None:
                    corrector.submit_polish(payload)
                if kind == "source":
                    with source_lock:
                        want_source["kind"] = payload
                if kind == "flush":
                    flush_req.set()
                if kind == "reset":
                    reset_req.set()
        except Exception:
            pass

    threading.Thread(target=watch_stdin, daemon=True).start()

    n_chunk = 0
    n_block = 0  # monotonic id for each committed block (for LLM correction)

    def emit_final(line, lt0, lt1):
        nonlocal n_block
        n_block += 1
        emit(
            {
                "type": "final",
                "id": n_block,
                "text": line,
                "t0": round(lt0, 2),
                "t1": round(lt1, 2),
            }
        )
        # LLM correction (Phase C): hand only *completed* sentences to the
        # worker. Enqueue and return — never block transcription on it.
        if corrector is not None and args.correct_sentences:
            corrector.submit(n_block, line)

    pass_hist = collections.deque(maxlen=12)
    consumed = 0.0                 # seconds of audio taken from the device
    t_start = time.perf_counter()
    latlog = LatencyLog(args.latency_csv)
    dump = AudioWriter(args.dump_audio)

    def switch_source(kind):
        """Swap the capture device without losing the session.

        Live captions read the speakers' loopback; dictation reads the
        microphone. Both feed the same 16 kHz stream, so only the device and
        the buffered audio have to change.
        """
        nonlocal stream, device, channels, rate, bytes_per_sec
        nonlocal consumed, t_start, source_kind
        try:
            dev = find_source(p, kind, args.device_index)
            st = open_input_stream(p, dev, frames, q)
        except Exception as exc:
            emit({"type": "status",
                  "msg": "%s unavailable (%s) — still listening to %s"
                         % (kind, exc, device["name"])})
            with source_lock:
                want_source["kind"] = source_kind
            return
        # Commit what the old device already produced: the switch is not a
        # reason to lose the last partial sentence.
        try:
            tail = streamer.flush()
            if tail:
                text = " ".join(w.text for w in tail)
                for line, lt0, lt1 in assembler.add_chunk(
                        text, tail[0].start, tail[-1].end):
                    emit_final(line, lt0, lt1)
        except Exception as exc:
            log("flush on source switch failed: %s" % exc)
        resume_at = streamer.buf_t0 + streamer.buf_secs
        streamer.reset(resume_at)
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        stream = st
        device = dev
        channels = int(dev["maxInputChannels"])
        rate = int(dev["defaultSampleRate"])
        bytes_per_sec = rate * channels * 2
        # Anything queued belongs to the old device (or to the few frames the
        # new one produced while the old stream was closing). Splicing the
        # two timelines together would garble the first caption.
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        consumed = resume_at
        t_start = time.perf_counter() - consumed
        pass_hist.clear()
        source_kind = kind
        log("Input switched to %s: %s" % (kind, dev["name"]))
        emit({"type": "status", "msg": "listening to %s" % dev["name"],
              "source": kind, "device": dev["name"]})

    def plan_step_secs():
        return plan_step(args.step_secs, args.step_headroom, pass_hist,
                         args.max_step_secs)

    def drop_backlog(excess):
        """Discard `excess` seconds of audio from the oldest end.

        The safety net under the adaptive step: if inference degrades (GPU
        contention, thermal throttling) faster than the step can absorb, the
        backlog is dropped rather than allowed to grow. Better to lose a
        moment of audio than to fall permanently behind the speaker.
        """
        nonlocal consumed
        if streamer.buf_secs >= excess:
            streamer._cut(streamer.buf_t0 + excess)
            consumed -= excess
            return excess
        excess -= streamer.buf_secs
        streamer.buf_t0 += streamer.buf_secs
        streamer.buf = np.zeros(0, dtype=np.float32)
        want = int(excess * bytes_per_sec)
        got = 0
        while got < want:
            try:
                got += len(q.get_nowait())
            except queue.Empty:
                break
        consumed -= got / bytes_per_sec
        return got / bytes_per_sec

    def collect_step(target_secs):
        """Drain audio until `target_secs` of it has arrived.

        A short read (idle loopback) is zero-padded, which is digital silence.
        """
        # Snap to whole frames. The adaptive planner returns a float, and
        # int(bytes_per_sec * target) is often ODD -- ~15% of adaptive targets.
        # An odd byte count then reaches np.frombuffer(dtype=int16) and raises
        # "buffer size must be a multiple of element size", killing the backend
        # after about a minute. A fixed step never produced odd counts, so this
        # only ever appeared once the step became adaptive.
        frame = frame_bytes_for(channels)
        want = int(bytes_per_sec * target_secs)
        want -= want % frame
        if want <= 0:
            want = frame
        out = bytearray()
        deadline = time.time() + target_secs + 2.0
        while len(out) < want:
            if stop_event.is_set():
                break
            if flush_req.is_set() or reset_req.is_set():
                break
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                out.extend(q.get(timeout=min(remaining, 0.05)))
            except queue.Empty:
                continue
        if len(out) < want:
            if not (flush_req.is_set() or reset_req.is_set()):
                # The flush will re-transcribe and cut whatever did arrive;
                # padding here would only add silence to that pass.
                out.extend(b"\x00" * (want - len(out)))
        if not out:
            return np.zeros(0, dtype=np.float32)
        return bytes_to_mono16k(bytes(out), channels, rate)

    corrector = None
    if args.correct_model:
        from correction import Corrector
        corrector = Corrector(args.correct_model, args.correct_threads)
        corrector.start()

    try:
        while not stop_event.is_set():
            # A clear arrived mid-collect: drop the pre-clear audio and the
            # pending sentence together, before any of it can be emitted.
            if reset_req.is_set():
                reset_req.clear()
                resume = streamer.buf_t0 + streamer.buf_secs
                streamer.reset(resume)
                assembler.reset()
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
                consumed = resume
                t_start = time.perf_counter() - consumed
                pass_hist.clear()
                emit({"type": "reset"})
                continue
            with source_lock:
                wanted = want_source["kind"]
            if wanted != source_kind:
                switch_source(wanted)
            target = plan_step_secs()
            step = collect_step(target)
            dump.add(step)
            if stop_event.is_set():
                break
            n_chunk += 1
            clock = time.strftime("%H:%M:%S")

            committed, unstable = streamer.feed(step)
            if streamer.pass_secs:
                pass_hist.append(streamer.pass_secs)

            # The TUI asked to edit/copy now: commit the words that were still
            # waiting for a second hypothesis, so the document it receives is
            # complete. `commit_now` also cuts the audio it just decoded.
            flush_now = flush_req.is_set()
            if flush_now:
                flush_req.clear()
                try:
                    committed = list(committed) + list(streamer.commit_now())
                except Exception as exc:
                    log("flush failed: %s" % exc)
                unstable = []

            # Drift guard. `consumed` is how much audio has been taken from
            # the device; wall time is how much should have arrived. Any
            # difference is backlog sitting in the queue or the buffer, and
            # it would otherwise become permanent caption lag.
            consumed += len(step) / TARGET_RATE
            behind = consumed - (time.perf_counter() - t_start)
            if behind > args.max_backlog:
                dropped = drop_backlog(behind - args.max_backlog)
                log("[%s step %d] backlog %.2fs > %.2fs, dropped %.2fs"
                    % (clock, n_chunk, behind, args.max_backlog, dropped))

            now = time.perf_counter()
            event = streamer.last_event or "none"
            for w in committed:
                latlog.add_word(n_chunk, w, streamer.word_latency(w, now),
                                streamer.pass_secs, target,
                                streamer.pass_buf_secs)
            if not committed:
                latlog.add_event(n_chunk, event, streamer.pass_secs, target,
                                 streamer.pass_buf_secs)

            log("[%s step %d] %-8s target %.2fs pass %.3fs buffer %.1fs "
                "backlog %.2fs -> +%d committed, %d unstable"
                % (clock, n_chunk, event, target, streamer.pass_secs,
                   streamer.pass_buf_secs, max(behind, 0.0),
                   len(committed), len(unstable)))

            # A burst is the visible symptom of a long freeze: audio piled up
            # behind a slow cycle and then arrived all at once. Flag it so the
            # log shows the cause instead of an unexplained wall of text.
            if len(committed) >= 20:
                log("[%s step %d] BURST %d words after target %.2fs "
                    "pass %.3fs" % (clock, n_chunk, len(committed), target,
                                    streamer.pass_secs))

            # A slow pass is the freeze itself. faster-whisper's temperature
            # fallback re-decodes a failing segment at up to six temperatures,
            # so a small buffer can take seconds and often yields nothing --
            # measured: 44 such passes and 99.5s of stall per 150s of audio.
            # `pass_buf_secs` distinguishes "big buffer, fairly priced" from
            # "small buffer, wildly overpriced", which is the actual bug.
            if streamer.pass_secs >= args.slow_pass_secs:
                log("[%s step %d] SLOWPASS %.2fs on %.1fs of audio%s "
                    "(event=%s)" % (clock, n_chunk, streamer.pass_secs,
                                    streamer.pass_buf_secs,
                                    ", no words" if streamer.pass_empty else "",
                                    event))

            if committed:
                text = " ".join(w.text for w in committed)
                ct0, ct1 = committed[0].start, committed[-1].end
                for line, lt0, lt1 in assembler.add_chunk(text, ct0, ct1):
                    emit_final(line, lt0, lt1)
            if flush_now:
                # The user asked for everything now. The assembler normally
                # holds a short sentence until 200 chars, so without this the
                # editor's text and the live draft would show the same
                # sentence twice.
                rest = assembler.flush()
                if rest:
                    emit_final(*rest)

            dtext, dt0 = compose_live_line(assembler, unstable)
            emit({"type": "draft", "text": dtext, "t0": round(dt0, 2),
                  "n": n_chunk})
            if flush_now:
                emit({"type": "flushed"})
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        stream.stop_stream()
        stream.close()
        # Recover words still waiting on agreement before the final flush.
        try:
            tail = streamer.flush()
            if tail:
                now = time.perf_counter()
                for w in tail:
                    latlog.add_word(n_chunk, w, streamer.word_latency(w, now),
                                    streamer.pass_secs, 0.0,
                                    streamer.pass_buf_secs)
                text = " ".join(w.text for w in tail)
                for line, lt0, lt1 in assembler.add_chunk(text, tail[0].start,
                                                          tail[-1].end):
                    emit_final(line, lt0, lt1)
        except Exception as exc:
            log("streamer flush failed: %s" % exc)
        log(latlog.summary())
        latlog.close()
        dump.close()
        tail_final = assembler.flush()
        if tail_final is not None:
            emit_final(*tail_final)
        if corrector is not None:
            corrector.stop()
        emit({"type": "status", "msg": "stopped"})


def main():
    # The protocol is UTF-8 JSON. Left alone, a pipe inherits the console code
    # page (cp1252 on Windows), so a device name or a "…" would be written as
    # invalid UTF-8 and arrive in the TUI as replacement characters.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="Earshot backend")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="continuous transcription loop")
    ap.add_argument("--device-index", type=int, default=None)
    ap.add_argument("--input", choices=["mic", "loopback"], default="loopback",
                    help="audio source: 'mic' for dictation, 'loopback' for "
                         "system audio (the TUI switches this at runtime)")
    ap.add_argument("--dump-secs", type=float, default=5.0)
    ap.add_argument("--dump-file", default="m1_probe.wav")
    ap.add_argument("--frames-per-buffer", type=int, default=512)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--step-secs", type=float, default=0.3,
                    help="minimum audio per cycle. The effective step grows "
                         "to match measured inference cost, because a pass "
                         "over ~0.5s of speech costs far more than a pass "
                         "over 0.3s; a fixed small step cannot keep up and "
                         "caption lag grows without bound")
    ap.add_argument("--step-headroom", type=float, default=1.25,
                    help="multiply the median pass time by this to size each "
                         "cycle, so consumption matches processing capacity")
    ap.add_argument("--max-step-secs", type=float, default=2.5,
                    help="hard ceiling on the adaptive step. Without it a "
                         "single slow pass ratchets the step up (measured: "
                         "18s pass -> 16.8s step -> 19.5s step), which freezes "
                         "captions and then flushes a burst")
    ap.add_argument("--latency-csv", default="",
                    help="write per-word spoken->committed latency as CSV")
    ap.add_argument("--dump-audio", default="",
                    help="write the captured mono 16k audio to this WAV, so a "
                         "session with a stall can be replayed offline")
    ap.add_argument("--slow-pass-secs", type=float, default=1.5,
                    help="log a SLOWPASS marker when a decode takes at least "
                         "this long. A slow pass on a *small* buffer is the "
                         "temperature-fallback signature, not a big buffer")
    ap.add_argument("--temp-fallback", choices=["full", "short", "off"],
                    default="off",
                    help="temperature fallback policy. off (default) passes "
                         "temperature 0, so a failing segment is never retried; "
                         "short caps retries at three; full is faster-whisper's "
                         "six-temperature default. Measured on 150s of "
                         "narration: off 1 slow pass / 4.8s stall, short 27 / "
                         "60.7s, full 55 / 220.8s -- at identical WER")
    ap.add_argument("--max-backlog", type=float, default=1.5,
                    help="seconds of audio allowed to queue before the "
                         "oldest is dropped. Safety net: bounds caption lag "
                         "even if inference degrades beyond the adaptive "
                         "step (GPU contention, thermal throttling)")
    ap.add_argument("--max-buf-secs", type=float, default=12.0,
                    help="buffer cap; exceeding it flushes the unstable tail. "
                         "Keeps inference cost bounded on long monologue")
    ap.add_argument("--pause-secs", type=float, default=0.2,
                    help="trailing silence that counts as a pause and commits "
                         "the tail. 0.2 measured best (WER 20.2%% on testclip); "
                         "0 makes every pass flush, which is the old "
                         "non-overlapping behaviour and scores 103%% WER")
    ap.add_argument("--stall-passes", type=int, default=6,
                    help="passes with no commit before force-flushing the "
                         "tail. A prefix match that never agrees blocks ALL "
                         "output, so this bounds the freeze")
    ap.add_argument("--min-speech-secs", type=float, default=0.10,
                    help="VAD gate: below this much speech in the buffer, "
                         "skip the model entirely and discard any tail. This "
                         "is what stops 'Thank you for watching' appearing "
                         "over music")
    ap.add_argument("--gate-min-buf-secs", type=float, default=0.9,
                    help="only judge the VAD gate on a buffer at least this "
                         "long, so short speech is never discarded")
    ap.add_argument("--prompt-chars", type=int, default=0,
                    help="committed characters fed back as initial_prompt "
                         "(0 = off; carries vocabulary across flushes but "
                         "risks hallucination)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--beam-size", type=int, default=1,
                    help="beam width. 1 (default) is greedy: on "
                         "call-quality audio hot spans it saves 0.6-0.9s "
                         "per 30s decode and cannot wander (docs/"
                         "latency-measurements.md mitigation sweep). Wider "
                         "beams recover a bad early token, which mattered "
                         "more on clean narration")
    ap.add_argument("--no-speech-threshold", type=float, default=0.6,
                    help="above this no-speech probability, read silence as silence")
    ap.add_argument("--hallucination-silence-threshold", type=float, default=2.0,
                    help="seconds; drop short low-confidence segments in silence")
    ap.add_argument("--compression-ratio-threshold", type=float, default=2.4,
                    help="gzip ratio above which a segment counts as repetitive")
    ap.add_argument("--log-prob-threshold", type=float, default=-1.0,
                    help="minimum average token logprob for a segment to survive")
    ap.add_argument("--block-chars", type=int, default=200,
                    help="sentence block size: emit a final per this many chars")
    ap.add_argument("--chain-chars", type=int, default=15,
                    help="trailing runes of the previous line chained into the next")
    ap.add_argument("--correct-model", default="",
                    help="path to a GGUF model; when set, each committed "
                         "sentence is polished by an LLM on a worker thread")
    ap.add_argument("--correct-threads", type=int, default=0,
                    help="threads for the LLM (0 = auto). The backend sets "
                         "OMP_NUM_THREADS=1 for resampling, so this must be set "
                         "explicitly or the model runs single-threaded")
    ap.add_argument("--correct-sentences", action="store_true",
                    help="patch each committed sentence as it lands (the old "
                         "behaviour). Off by default: voice-to-text polishes the "
                         "whole document instead, and doing both is twice the "
                         "work for one result")
    ap.add_argument("--vad-min-silence-ms", type=int, default=150,
                    help="Silero VAD silence needed to split a segment")
    ap.add_argument("--vad-pad-ms", type=int, default=50,
                    help="Silero VAD padding around speech")
    args = ap.parse_args()

    import pyaudiowpatch as pyaudio

    try:
        with pyaudio.PyAudio() as p:
            if args.list_devices:
                cmd_list_devices(p)
            elif args.live:
                cmd_live(p, args)
            else:
                cmd_dump(p, args)
    except Exception as e:  # noqa: BLE001 — report as JSON, then exit nonzero
        emit({"type": "error", "msg": str(e)})
        # Full traceback to stderr, which the Go frontend copies into
        # .backend.log. A bare one-line message made a real crash
        # ("buffer size must be a multiple of element size") undiagnosable.
        traceback.print_exc()
        log("ERROR: %s" % e)
        sys.exit(2)


if __name__ == "__main__":
    main()
