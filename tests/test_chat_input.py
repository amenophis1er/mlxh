import asyncio

import pytest

prompt_toolkit = pytest.importorskip("prompt_toolkit")
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

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
