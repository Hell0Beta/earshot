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
import faulthandler
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
        """Emit the remainder, however short. Returns (line, t0, t1) or None."""
        if not self.buf.strip():
            return None
        self.buf = self.buf.strip()
        out = self._emit_head(self.buf)
        self.buf = ""
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

def plan_step(floor_secs, headroom, pass_hist):
    """How much audio the next cycle may consume.

    Sized from the *measured* cost of recent passes, not from a fixed number.
    One pass over ~0.5s of real speech costs far more than a pass over 0.3s
    (measured: 111 of 131 real passes exceeded 100ms, p90 ~590ms), so a fixed
    small step cannot keep up -- every cycle overruns the audio it represents,
    the queue grows, and caption lag increases without bound. That is the
    accumulating-latency bug.

    Consuming at least the pass cost is what holds drift at zero: the pipeline
    then runs exactly as fast as the machine allows instead of accumulating
    debt. p90 (not max) so a single slow spike does not inflate every later
    step. `floor_secs` keeps the caller's minimum cadence.
    """
    if not pass_hist:
        return floor_secs
    ordered = sorted(pass_hist)
    idx = min(len(ordered) - 1, max(0, int(round(0.9 * len(ordered))) - 1))
    p90 = ordered[idx]
    return max(floor_secs, p90 * headroom)


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

    step_secs = args.step_secs
    assembler = SentenceAssembler(args.block_chars, args.chain_chars)

    from streaming import RollingStreamer

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
    )

    emit(
        {
            "type": "status",
            "msg": "live",
            "device": device["name"],
            "step_secs": step_secs,
            "block_chars": args.block_chars,
            "beam_size": args.beam_size,
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
        if corrector is not None:
            corrector.submit(n_block, line)

    pass_hist = collections.deque(maxlen=12)
    consumed = 0.0                 # seconds of audio taken from the device

    def plan_step_secs():
        return plan_step(args.step_secs, args.step_headroom, pass_hist)

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
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                out.extend(q.get(timeout=min(remaining, 0.05)))
            except queue.Empty:
                continue
        if len(out) < want:
            out.extend(b"\x00" * (want - len(out)))
        return bytes_to_mono16k(bytes(out), channels, rate)

    corrector = None
    if args.correct_model:
        from correction import Corrector
        corrector = Corrector(args.correct_model, args.correct_threads)
        corrector.start()

    try:
        t_start = time.perf_counter()
        while not stop_event.is_set():
            target = plan_step_secs()
            step = collect_step(target)
            if stop_event.is_set():
                break
            n_chunk += 1
            clock = time.strftime("%H:%M:%S")

            committed, unstable = streamer.feed(step)
            if streamer.pass_secs:
                pass_hist.append(streamer.pass_secs)

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

            log("[%s step %d] target %.2fs pass %.3fs buffer %.1fs "
                "backlog %.2fs -> +%d committed, %d unstable"
                % (clock, n_chunk, target, streamer.pass_secs,
                   streamer.buf_secs, max(behind, 0.0),
                   len(committed), len(unstable)))

            if committed:
                text = " ".join(w.text for w in committed)
                ct0, ct1 = committed[0].start, committed[-1].end
                for line, lt0, lt1 in assembler.add_chunk(text, ct0, ct1):
                    emit_final(line, lt0, lt1)

            dtext, dt0 = compose_live_line(assembler, unstable)
            emit({"type": "draft", "text": dtext, "t0": round(dt0, 2),
                  "n": n_chunk})
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        stream.stop_stream()
        stream.close()
        # Recover words still waiting on agreement before the final flush.
        try:
            tail = streamer.flush()
            if tail:
                text = " ".join(w.text for w in tail)
                for line, lt0, lt1 in assembler.add_chunk(text, tail[0].start,
                                                          tail[-1].end):
                    emit_final(line, lt0, lt1)
        except Exception as exc:
            log("streamer flush failed: %s" % exc)
        tail_final = assembler.flush()
        if tail_final is not None:
            emit_final(*tail_final)
        if corrector is not None:
            corrector.stop()
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
    ap.add_argument("--step-secs", type=float, default=0.3,
                    help="minimum audio per cycle. The effective step grows "
                         "to match measured inference cost, because a pass "
                         "over ~0.5s of speech costs far more than a pass "
                         "over 0.3s; a fixed small step cannot keep up and "
                         "caption lag grows without bound")
    ap.add_argument("--step-headroom", type=float, default=1.25,
                    help="multiply the p90 pass time by this to size each "
                         "cycle, so consumption matches processing capacity")
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
    ap.add_argument("--beam-size", type=int, default=5,
                    help="beam width; 1 is greedy and cannot recover a bad "
                         "early token")
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
