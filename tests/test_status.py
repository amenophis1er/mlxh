import json
from types import SimpleNamespace as NS

import pytest

import mlxh.cli as cli


PAYLOAD = {
    "model": "bonsai2",
    "settings": {"max_queued": 4},
    "mlx": {
        "active_memory_bytes": 13_200_000_000,
        "cache_memory_bytes": 1_200_000_000,
        "last_peak_memory_bytes": 13_900_000_000,
    },
    "runtime": {
        "engine_version": 1,
        "uptime_s": 3601,
        "pid": 12345,
        "ready": True,
        "busy": True,
        "queue_depth": 0,
        "requests": 42,
        "prompt_tokens": 98_304,
        "tokens_generated": 183_456,
        "mlx_version": "0.32.0",
    },
}


def test_status_table(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_fetch_info", lambda _port: PAYLOAD)
    cli.cmd_status(NS(json=False))
    out = capsys.readouterr().out
    for value in ("MODEL", "bonsai2", "BUSY", "98.3k", "183k",
                  "13.2 GB", "1.2 GB", "13.9 GB"):
        assert value in out


@pytest.mark.parametrize(
    ("ready", "busy", "state"),
    [(False, False, "LOADING"), (True, True, "BUSY"), (True, False, "IDLE")],
)
def test_status_state(monkeypatch, capsys, ready, busy, state):
    payload = {**PAYLOAD, "runtime": {**PAYLOAD["runtime"],
                                      "ready": ready, "busy": busy}}
    monkeypatch.setattr(cli, "_fetch_info", lambda _port: payload)
    cli.cmd_status(NS(json=False))
    assert state in capsys.readouterr().out


def test_status_missing_and_null_values(monkeypatch, capsys):
    payload = {
        "model": "loading",
        "settings": {},
        "mlx": {"active_memory_bytes": 400_000_000,
                "cache_memory_bytes": None},
        "runtime": {"ready": False, "busy": False, "queue_depth": 0},
    }
    monkeypatch.setattr(cli, "_fetch_info", lambda _port: payload)
    cli.cmd_status(NS(json=False))
    out = capsys.readouterr().out
    assert "LOADING" in out
    assert out.count("—") >= 5


def test_status_old_server_has_unknown_state(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_fetch_info", lambda _port: {"model": "old", "settings": {}}
    )
    cli.cmd_status(NS(json=False))
    out = capsys.readouterr().out
    assert "old" in out and "—" in out
    assert "LOADING" not in out
    assert "older mlxh server" in out


def test_status_json_preserves_payload_and_adds_port(monkeypatch, capsys):
    payload = {**PAYLOAD, "port": 9999,
               "mlx": {**PAYLOAD["mlx"], "cache_memory_bytes": None}}
    monkeypatch.setattr(cli, "_fetch_info", lambda _port: payload)
    cli.cmd_status(NS(json=True))
    result = json.loads(capsys.readouterr().out)
    assert result == {**payload, "port": cli.load_config()["port"]}
    assert result["mlx"]["cache_memory_bytes"] is None


@pytest.mark.parametrize("json_output", [False, True])
def test_status_failure_is_stderr_only(monkeypatch, capsys, json_output):
    def fail(_port):
        raise ValueError("bad response")

    monkeypatch.setattr(cli, "_fetch_info", fail)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_status(NS(json=json_output))
    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert captured.out == ""
    assert "no mlxh server running" in captured.err
