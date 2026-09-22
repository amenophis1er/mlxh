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
