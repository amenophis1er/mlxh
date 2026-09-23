import asyncio
import subprocess
import urllib.request
from pathlib import Path

import pytest

prompt_toolkit = pytest.importorskip("prompt_toolkit")
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

import mlxh.chat_cli as chat_cli
from mlxh.chat_cli import _chat_session


def test_multiline_bracketed_paste_waits_for_explicit_enter(tmp_path):
    with create_pipe_input() as pipe:
        session = _chat_session(
            tmp_path / "history", ["/exit"], input=pipe, output=DummyOutput()
        )

        async def exercise():
            task = asyncio.create_task(
                session.prompt_async("> ", prompt_continuation="... ")
            )
            await asyncio.sleep(0)
            pipe.send_text("\x1b[200~first line\nsecond line\x1b[201~")
            await asyncio.sleep(0.05)
            assert not task.done(), "the paste submitted without an explicit Enter"
            pipe.send_text("\r")
            return await asyncio.wait_for(task, timeout=1)

        assert asyncio.run(exercise()) == "first line\nsecond line"


def test_slash_command_completion(tmp_path):
    session = _chat_session(tmp_path / "history", ["/exit", "/reset"])
    document = prompt_toolkit.document.Document("/ex", cursor_position=3)
    completions = list(session.completer.get_completions(document, None))
    assert [item.text for item in completions] == ["/exit"]


def test_existing_readline_history_remains_available(tmp_path):
    history = tmp_path / "history"
    history.write_text("older question\nnewer question\n")
    session = _chat_session(history, [])
    assert list(session.history.load_history_strings()) == [
        "newer question", "older question"
    ]


def test_prepare_image_accepts_path_url_and_clipboard(tmp_path, monkeypatch):
    local = tmp_path / "image with spaces.png"
    local.write_bytes(b"image")
    monkeypatch.setattr(chat_cli, "_validate_image", lambda _path: None)
    monkeypatch.setattr(chat_cli, "_download_image", lambda url: "/tmp/url.png")
    monkeypatch.setattr(chat_cli, "_clipboard_image", lambda: "/tmp/clipboard.png")

    assert chat_cli._prepare_image(str(local)) == (str(local), False)
    assert chat_cli._prepare_image(f'"{local}"') == (str(local), False)
    assert chat_cli._prepare_image("https://example.com/a.png") == (
        "/tmp/url.png", True
    )
    assert chat_cli._prepare_image() == ("/tmp/clipboard.png", True)
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        chat_cli._prepare_image("file:///tmp/a.png")


def test_clipboard_image_is_captured_to_a_temporary_file(monkeypatch):
    def run(args, **_kwargs):
        Path(args[-1]).write_bytes(b"clipboard image")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(chat_cli, "_validate_image", lambda _path: None)
    path = Path(chat_cli._clipboard_image())
    try:
        assert path.suffix == ".png"
        assert path.read_bytes() == b"clipboard image"
    finally:
        path.unlink(missing_ok=True)


def test_download_image_is_bounded_and_validated(monkeypatch):
    class Response:
        headers = {"Content-Type": "image/png", "Content-Length": "5"}

        def __init__(self):
            self.chunks = iter((b"image", b""))
            self.closed = False

        def geturl(self):
            return "https://cdn.example.com/image.png"

        def read(self, _size):
            return next(self.chunks)

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(chat_cli, "_validate_image", lambda _path: None)
    path = Path(chat_cli._download_image("https://example.com/image.png"))
    try:
        assert path.read_bytes() == b"image"
        assert path.suffix == ".png"
        assert response.closed
    finally:
        path.unlink(missing_ok=True)


def test_download_image_stops_when_stream_exceeds_limit(monkeypatch):
    class Response:
        headers = {"Content-Type": "image/png"}

        def geturl(self):
            return "https://example.com/image.png"

        def read(self, _size):
            return b"12345"

        def close(self):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(chat_cli, "MAX_IMAGE_DOWNLOAD", 4)
    with pytest.raises(ValueError, match="larger than"):
        chat_cli._download_image("https://example.com/image.png")
