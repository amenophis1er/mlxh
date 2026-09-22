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


import io


def renderer():
    buf = io.StringIO()
    r = ui.StreamRenderer(buf)
    r.enabled = True  # force styling despite non-tty test buffer
    return r, buf


def test_stream_passthrough_when_disabled():
    buf = io.StringIO()
    r = ui.StreamRenderer(buf)  # StringIO is not a tty -> disabled
    r.feed("**raw** `text`")
    r.finish()
    assert buf.getvalue() == "**raw** `text`"


def test_stream_bold_split_across_chunks():
    r, buf = renderer()
    r.feed("**bo")
    r.feed("ld** x")
    r.finish()
    assert buf.getvalue() == "\x1b[1mbold\x1b[22m x"


def test_stream_marker_split_mid_pair():
    r, buf = renderer()
    r.feed("*")
    r.feed("*hi**")
    r.finish()
    assert buf.getvalue() == "\x1b[1mhi\x1b[22m"


def test_stream_inline_code():
    r, buf = renderer()
    r.feed("a `b` c")
    r.finish()
    assert buf.getvalue() == "a \x1b[36mb\x1b[39m c"


def test_stream_fence_dimmed_and_literal_inside():
    r, buf = renderer()
    r.feed("```py\nx = '**not bold**'\n```\ndone")
    r.finish()
    out = buf.getvalue()
    assert out.startswith("\x1b[2m```py")
    assert "**not bold**" in out  # no styling inside the fence
    assert "\x1b[22m" in out and out.endswith("done")


def test_stream_fence_marker_split():
    r, buf = renderer()
    r.feed("`")
    r.feed("``\ncode\n```\n")
    r.finish()
    assert buf.getvalue().startswith("\x1b[2m```")


def test_stream_heading():
    r, buf = renderer()
    r.feed("# Title\nbody")
    r.finish()
    assert buf.getvalue() == "\x1b[1m# Title\x1b[22m\nbody"


def test_stream_unclosed_style_reset_on_finish():
    r, buf = renderer()
    r.feed("**oops")
    r.finish()
    assert buf.getvalue().endswith("\x1b[0m")


def test_stream_reasoning_dimmed_until_close():
    buf = io.StringIO()
    r = ui.StreamRenderer.__new__(ui.StreamRenderer)
    r.out, r.enabled, r.carry = buf, True, ""
    r.bold = r.code = r.fence = r.heading = False
    r.line_start, r.reasoning, r._rbuf = True, True, ""
    buf.write(r.DIM)  # what __init__ does when think_open
    r.feed("step one, step two")
    r.feed("</think>\n\nThe answer is 4.")
    r.finish()
    out = buf.getvalue()
    assert out.startswith("\x1b[2m")
    assert "step one" in out and "</think>" not in out
    assert out.endswith("The answer is 4.")


def test_stream_reasoning_tag_split_across_chunks():
    r, buf = renderer()
    r.reasoning, r._rbuf = True, ""
    r.feed("hmm</thi")
    r.feed("nk>\nanswer")
    r.finish()
    out = buf.getvalue()
    assert "</think>" not in out and out.endswith("answer")


def test_stream_reasoning_never_closes_resets():
    r, buf = renderer()
    r.reasoning, r._rbuf = True, ""
    r.feed("endless pondering")
    r.finish()
    assert buf.getvalue().endswith("\x1b[0m")
