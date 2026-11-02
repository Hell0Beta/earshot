"""Unit tests for the backend's stdin command protocol.

The TUI talks to the backend over stdin: plain "quit", a JSON polish
request, or a source switch. Parsing is separated from the reader thread so it can be tested
without a microphone, a model, or a live stream.

Run: ..\\.venv\\Scripts\\python.exe backend\\test_captions.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import captions


def test_quit_is_recognised():
    assert captions.parse_command("quit\n") == ("quit", None)
    assert captions.parse_command("QUIT") == ("quit", None)


def test_polish_carries_the_document():
    line = json.dumps({"cmd": "polish", "text": "hello there"})
    assert captions.parse_command(line) == ("polish", "hello there")


def test_missing_text_is_an_empty_document():
    assert captions.parse_command('{"cmd":"polish"}') == ("polish", "")


def test_unknown_commands_are_ignored():
    for line in ("", "   ", "garbage", '{"cmd":"stop"}', "[1,2,3]", "null", "42"):
        assert captions.parse_command(line) == (None, None), line


def test_source_switch_is_recognised():
    line = json.dumps({"cmd": "source", "kind": "mic"})
    assert captions.parse_command(line) == ("source", "mic")
    line = json.dumps({"cmd": "source", "kind": "loopback"})
    assert captions.parse_command(line) == ("source", "loopback")


def test_flush_is_recognised():
    assert captions.parse_command('{"cmd":"flush"}') == ("flush", None)


def test_reset_is_recognised():
    assert captions.parse_command('{"cmd":"reset"}') == ("reset", None)


def test_unknown_source_is_ignored():
    for kind in ("speakers", "", None, 3):
        line = json.dumps({"cmd": "source", "kind": kind})
        assert captions.parse_command(line) == (None, None), kind


class FakeAudio:
    """Just enough of PyAudioWPatch to test device selection."""

    def __init__(self, devices, default_input):
        self.devices = devices
        self.default_input = default_input

    def get_device_info_by_index(self, i):
        return self.devices[i]

    def get_device_info_generator(self):
        return iter(self.devices)

    def get_default_input_device_info(self):
        if self.default_input is None:
            raise OSError("no default input")
        return self.devices[self.default_input]

    def get_host_api_info_by_type(self, kind):
        raise OSError("WASAPI unavailable in this fake")


def dev(index, name, inputs, loopback=False):
    return {"index": index, "name": name, "maxInputChannels": inputs,
            "maxOutputChannels": 2, "defaultSampleRate": 48000,
            "isLoopbackDevice": loopback}


def test_mic_ignores_a_loopback_default_input():
    # pyaudiowpatch can report a loopback endpoint as the default input; that
    # is the bug that made dictation transcribe the speakers.
    p = FakeAudio([dev(0, "Speakers [Loopback]", 2, True),
                   dev(1, "Microphone Array", 2)], default_input=0)
    assert captions.find_mic(p)["index"] == 1


def test_mic_prefers_the_real_default_input():
    p = FakeAudio([dev(0, "Speakers [Loopback]", 2, True),
                   dev(1, "Microphone Array", 2)], default_input=1)
    assert captions.find_mic(p)["index"] == 1


def test_mic_never_returns_a_loopback_device():
    p = FakeAudio([dev(0, "Speakers [Loopback]", 2, True),
                   dev(1, "Headset", 1)], default_input=0)
    info = captions.find_mic(p)
    assert not info["isLoopbackDevice"]
    assert info["index"] == 1


def test_mic_rejects_an_explicit_loopback_index():
    p = FakeAudio([dev(0, "Speakers [Loopback]", 2, True),
                   dev(1, "Microphone Array", 2)], default_input=1)
    try:
        captions.find_mic(p, device_index=0)
    except RuntimeError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("a loopback index must be rejected")


def test_mic_without_any_input_device_raises():
    p = FakeAudio([dev(0, "Speakers [Loopback]", 2, True)], default_input=None)
    try:
        captions.find_mic(p)
    except RuntimeError as exc:
        assert "No microphone" in str(exc)
    else:
        raise AssertionError("no input device must raise")


def test_source_dispatch():
    p = FakeAudio([dev(0, "Speakers [Loopback]", 2, True),
                   dev(1, "Microphone Array", 2)], default_input=1)
    assert captions.find_source(p, "mic")["index"] == 1
    # find_loopback needs WASAPI helpers; it is covered by the live probe.


def test_bad_json_does_not_raise():
    assert captions.parse_command('{"cmd":') == (None, None)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok:", t.__name__)
    print("ALL %d PASSED" % len(tests))
