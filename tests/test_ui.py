import io
import re

import pytest

import mlxh.ui as ui


def test_numbered_fallback_reprompts(monkeypatch):
    answers = iter(["junk", "2"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert ui._select_numbered("t", ["a", "b"]) == 1


def test_numbered_accepts_name(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "a")
    assert ui._select_numbered("t", ["a", "b"]) == 0


def test_select_falls_back_without_tty(monkeypatch):
    # pytest's captured stdin has no usable fd: termios setup fails -> fallback
    monkeypatch.setattr("builtins.input", lambda *a: "1")
    assert ui.select("t", ["a", "b"]) == 0


def test_fail_exits_with_block(capsys):
    with pytest.raises(SystemExit) as e:
        ui.fail("boom", "a detail", hint="try x")
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "boom" in err and "a detail" in err and "try x" in err


def test_colors_off_when_piped(capsys):
    assert ui.dim("x") == "x"  # captured stdout is not a tty


def renderer():
    buf = io.StringIO()
    r = ui.StreamRenderer(buf, force_terminal=True)
    return r, buf


def test_stream_passthrough_when_disabled():
    buf = io.StringIO()
    r = ui.StreamRenderer(buf)  # StringIO is not a tty -> disabled
    r.feed("**raw** `text`")
    r.finish()
    assert buf.getvalue() == "**raw** `text`"


def test_stream_preserves_markdown_split_across_chunks():
    r, buf = renderer()
    r.feed("**bo")
    r.feed("ld** and `code`")
    assert r.markdown_text == "**bold** and `code`"
    r.finish()
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", buf.getvalue())
    assert "bold" in plain and "code" in plain


def test_stream_renders_structural_markdown():
    r, buf = renderer()
    r.feed(
        "# Title\n\n- one\n- two\n\n"
        "| A | B |\n|---|---|\n| x | y |\n\n"
        "```python\nprint(1)\n```"
    )
    r.finish()
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", buf.getvalue())
    assert "# Title" not in plain
    assert " • one" in plain and " • two" in plain
    assert "A  B" in plain and "x  y" in plain
    assert "```" not in plain and "print(1)" in plain


def test_stream_only_commits_blank_lines_outside_fences():
    text = "```text\nfirst\n\nsecond\n```\n\nafter"
    boundary = ui.StreamRenderer._block_boundary(text)
    assert text[:boundary] == "```text\nfirst\n\nsecond\n```\n\n"


def test_stream_incomplete_markdown_finishes_cleanly():
    r, buf = renderer()
    r.feed("**oops")
    r.finish()
    assert r._live is None
    assert "oops" in buf.getvalue()


def test_stream_reasoning_dimmed_until_close():
    buf = io.StringIO()
    r = ui.StreamRenderer(buf, think_open=True, force_terminal=True)
    r.feed("step one, step two")
    r.feed("</think>\n\nThe answer is 4.")
    r.finish()
    out = buf.getvalue()
    assert "step one" in out and "</think>" not in out
    assert r.markdown_text == "The answer is 4."


def test_stream_reasoning_tag_split_across_chunks():
    buf = io.StringIO()
    r = ui.StreamRenderer(buf, think_open=True, force_terminal=True)
    r.feed("hmm</thi")
    r.feed("nk>\nanswer")
    r.finish()
    assert "</think>" not in buf.getvalue()
    assert r.markdown_text == "answer"


def test_stream_reasoning_never_closes_resets():
    buf = io.StringIO()
    r = ui.StreamRenderer(buf, think_open=True, force_terminal=True)
    r.feed("endless pondering")
    r.finish()
    assert not r.reasoning and r._live is None
    assert "endless pondering" in buf.getvalue()
