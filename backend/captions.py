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
import faulthandler
import json
import math
import os
import queue
import sys
import threading
import time
import wave

import numpy as np
from scipy.signal import resample_poly  # eager: pay the ~1.5s import at startup,
                                        # never silently mid-stream (M3 stall)

TARGET_RATE = 16000
DEFAULT_MODEL_DIR = os.environ.get("EARSHOT_MODEL_DIR", r"D:\earshot\models")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


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


def resample_to_16k(mono, src_rate):
    """Mono float32 at src_rate -> float32 at 16 kHz."""
    if src_rate == TARGET_RATE:
        return mono.astype(np.float32)
    g = math.gcd(TARGET_RATE, src_rate)
    up, down = TARGET_RATE // g, src_rate // g
    return resample_poly(mono, up, down).astype(np.float32)


def bytes_to_mono16k(raw, channels, src_rate):
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    audio = audio[: (len(audio) // channels) * channels].reshape(-1, channels)
    return resample_to_16k(audio.mean(axis=1), src_rate)


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


def open_loopback_stream(p, device, frames, q):
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
    stream = open_loopback_stream(p, device, frames, q)
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


def collect_window(q, n_bytes, timeout, stop_event=None):
    """Drain up to n_bytes from q within timeout; zero-pad shortfall.

    The loopback delivers nothing while idle, so wall-clock (not callbacks)
    drives chunking. Missing bytes == digital silence.
    """
    out = bytearray()
    deadline = time.time() + timeout
    while len(out) < n_bytes:
        if stop_event is not None and stop_event.is_set():
            break
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            out.extend(q.get(timeout=min(remaining, 0.25)))
        except queue.Empty:
            continue
    if len(out) < n_bytes:
        out.extend(b"\x00" * (n_bytes - len(out)))
    return bytes(out)


class SentenceAssembler:
    """Accumulates chunk texts into fixed-size display blocks.

    Length-driven, never time-driven: the first line takes up to
    `block_chars` runes; every following line chains the last `chain_chars`
    runes of the previous line plus fresh text up to
    `block_chars - chain_chars - 1` (the -1 is the joining space), so every
    emitted line stays within `block_chars`. Cuts land on word boundaries
    (hard cut only for a spaceless run longer than the limit); the remainder
    stays buffered. `flush()` emits whatever is left (e.g. on session stop).
    buf_t0 is the start of the oldest unconsumed text; per-char timing isn't
    tracked, so line timestamps overlap slightly by design.
    """

    def __init__(self, block_chars=200, chain_chars=15):
        self.block_chars = block_chars
        self.chain_chars = chain_chars
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
        return word.lower().strip(",.!?;:")

    @classmethod
    def _collapse_repeats(cls, text):
        """Drop immediately-repeated words ("is is", "from from").

        Short windows make Whisper occasionally repeat a token; adjacent exact
        duplicates are never legitimate speech.
        """
        out = []
        for w in text.split():
            if out and cls._norm(out[-1]) == cls._norm(w):
                continue
            out.append(w)
        return " ".join(out)

    def _trim_overlap(self, text):
        """Strip words the new chunk repeats from the buffer tail.

        Consecutive windows overlap by design, so chunk N+1 usually restates
        the tail of chunk N. Without trimming, the live line stutters. The
        threshold is 2 words: window overlaps of 0.2s routinely restate one
        or two words, and at a known boundary a 2-word repeat is not speech.
        """
        tail_words = self.buf.split()[-12:]
        new_words = text.split()
        top = min(len(tail_words), len(new_words))
        for n in range(top, 1, -1):
            if all(self._norm(a) == self._norm(b)
                   for a, b in zip(tail_words[-n:], new_words[:n])):
                return " ".join(new_words[n:])
        return text

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
        text = self._collapse_repeats(self._trim_overlap(text.strip()))
        # Cross-boundary single repeat: the buffer ends with a word the new
        # chunk restates ("...directly from" + "from the speakers").
        if text and self.buf:
            buf_words = self.buf.split()
            if buf_words and self._norm(buf_words[-1]) == self._norm(text.split()[0]):
                text = " ".join(text.split()[1:])
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
        """Emit the remainder, however short. Returns (line, t0, t1) or None."""
        if not self.buf.strip():
            return None
        self.buf = self.buf.strip()
        out = self._emit_head(self.buf)
        self.buf = ""
        return out


def cmd_live(p, args):
    device = find_loopback(p, args.device_index)
    channels = int(device["maxInputChannels"])
    rate = int(device["defaultSampleRate"])
    frames = int(args.frames_per_buffer)
    bytes_per_sec = rate * channels * 2  # int16

    emit({"type": "status", "msg": "loading model (one-time download on first run)…"})
    model = load_model(args)

    q = queue.Queue()
    stream = open_loopback_stream(p, device, frames, q)
    emit(
        {
            "type": "status",
            "msg": "live",
            "device": device["name"],
            "chunk_secs": args.chunk_secs,
            "overlap_secs": args.overlap_secs,
        }
    )
    log("Live transcription started — Ctrl+C to stop.")
    # One-shot watchdog: if the process ever wedges, dump every thread's stack
    # to stderr once so the frozen frame is on record (M3 stall forensics).
    faulthandler.dump_traceback_later(120, repeat=False)

    stop_event = threading.Event()

    def watch_stdin():
        try:
            for line in sys.stdin:
                if line.strip().lower() == "quit":
                    stop_event.set()
                    break
        except Exception:
            pass

    threading.Thread(target=watch_stdin, daemon=True).start()

    assembler = SentenceAssembler(args.block_chars, args.chain_chars)
    new_secs = args.chunk_secs - args.overlap_secs
    new_bytes = int(bytes_per_sec * new_secs)
    overlap_n = int(TARGET_RATE * args.overlap_secs)
    tail = np.zeros(overlap_n, dtype=np.float32)
    t_start = time.time()  # session wall clock, for line timestamps
    n_chunk = 0
    try:
        while not stop_event.is_set():
            raw = collect_window(q, new_bytes, timeout=new_secs + 2.0,
                                 stop_event=stop_event)
            if stop_event.is_set():
                break
            fresh = bytes_to_mono16k(raw, channels, rate)
            if len(fresh) < len(tail):
                fresh = np.pad(fresh, (0, len(tail) - len(fresh)))
            else:
                fresh = fresh[: int(TARGET_RATE * new_secs)]
            window = np.concatenate([tail, fresh])
            tail = window[-overlap_n:] if overlap_n else np.zeros(0, dtype=np.float32)
            # Wall-clock session time: monotonic and drift-free (the old
            # accumulated t_base could report 00:00 for late lines).
            t0 = max(time.time() - t_start - args.chunk_secs, 0.0)
            n_chunk += 1

            peak = float(np.abs(window).max())
            nbuf = len(assembler.buf)
            clock = time.strftime("%H:%M:%S")
            if peak < args.silence_threshold:
                log("[%s chunk %d] silent (peak=%.4f, buf=%d), skipped"
                    % (clock, n_chunk, peak, nbuf))
            else:
                log("[%s chunk %d] transcribing %.1fs (peak=%.3f, buf=%d)…"
                    % (clock, n_chunk, len(window) / TARGET_RATE, peak, nbuf))
                t_inf = time.time()
                segments, _info = model.transcribe(
                    window,
                    language=args.language,
                    beam_size=1,
                    best_of=1,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=args.vad_min_silence_ms,
                        speech_pad_ms=args.vad_pad_ms,
                    ),
                    condition_on_previous_text=False,
                )
                log("[chunk %d] inference %.2fs" % (n_chunk, time.time() - t_inf))
                segs = [s for s in segments if s.text.strip()]
                if segs:
                    chunk_text = " ".join(s.text.strip() for s in segs)
                    ct0 = max(t0 + min(s.start for s in segs), 0.0)
                    ct1 = max(t0 + max(s.end for s in segs), 0.0)
                    for line, lt0, lt1 in assembler.add_chunk(chunk_text, ct0, ct1):
                        emit(
                            {
                                "type": "final",
                                "text": line,
                                "t0": round(lt0, 2),
                                "t1": round(lt1, 2),
                            }
                        )
            dtext, dt0 = assembler.draft()
            emit({"type": "draft", "text": dtext, "t0": round(dt0, 2),
                  "n": n_chunk})
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        stream.stop_stream()
        stream.close()
        tail_final = assembler.flush()
        if tail_final is not None:
            line, lt0, lt1 = tail_final
            emit(
                {
                    "type": "final",
                    "text": line,
                    "t0": round(lt0, 2),
                    "t1": round(lt1, 2),
                }
            )
        emit({"type": "status", "msg": "stopped"})


def main():
    ap = argparse.ArgumentParser(description="Earshot backend")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="continuous transcription loop")
    ap.add_argument("--device-index", type=int, default=None)
    ap.add_argument("--dump-secs", type=float, default=5.0)
    ap.add_argument("--dump-file", default="m1_probe.wav")
    ap.add_argument("--frames-per-buffer", type=int, default=512)
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--chunk-secs", type=float, default=1.0)
    ap.add_argument("--overlap-secs", type=float, default=0.2)
    ap.add_argument("--language", default="en")
    ap.add_argument("--silence-threshold", type=float, default=0.005)
    ap.add_argument("--block-chars", type=int, default=200,
                    help="sentence block size: emit a final per this many chars")
    ap.add_argument("--chain-chars", type=int, default=15,
                    help="trailing runes of the previous line chained into the next")
    ap.add_argument("--vad-min-silence-ms", type=int, default=150,
                    help="Silero VAD silence needed to split (keep < chunk)")
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
        log("ERROR: %s" % e)
        sys.exit(2)


if __name__ == "__main__":
    main()
