import os
import plistlib
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

import mlxh.cli as cli


def test_service_plist_round_trip_and_forces_localhost():
    xml = cli.service_plist(
        "/Users/a&b/.local/bin/mlxh",
        'model<&"name',
        "/custom/home/service.log",
        "/custom/home",
    )
    data = plistlib.loads(xml.encode())
    assert data["Label"] == "com.mlxh.serve"
    assert data["ProgramArguments"] == [
        "/Users/a&b/.local/bin/mlxh", "serve", 'model<&"name',
        "--host", "127.0.0.1",
    ]
    assert data["KeepAlive"] is True
    assert data["ThrottleInterval"] == 60
    assert data["EnvironmentVariables"]["MLXH_HOME"] == "/custom/home"
    assert data["StandardOutPath"] == "/custom/home/service.log"
    assert data["StandardErrorPath"] == "/custom/home/service.log"
    assert "--port" not in data["ProgramArguments"]


@pytest.fixture
def service_env(tmp_path, monkeypatch):
    home = tmp_path / "mlxh-home"
    models = home / "models"
    model = models / "bonsai2"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    config = home / "config.json"
    config.write_text(
        f'{{"service_model": "bonsai2", "port": 1060, '
        f'"models_dir": "{models}"}}'
    )
    plist = tmp_path / "Library" / "LaunchAgents" / "com.mlxh.serve.plist"
    monkeypatch.setattr(cli, "HOME", home)
    monkeypatch.setattr(cli, "CONFIG", config)
    monkeypatch.setattr(cli, "_service_plist_path", lambda: plist)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/opt/bin/mlxh")
    monkeypatch.delenv("MLXH_MODELS_DIR", raising=False)
    return home, plist


def test_service_installs_model_manager_without_model_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "HOME", tmp_path)
    monkeypatch.setattr(cli, "CONFIG", tmp_path / "missing.json")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/opt/bin/mlxh")
    monkeypatch.setattr(cli, "_service_loaded", lambda: False)
    monkeypatch.setattr(cli, "_port_in_use", lambda _port: False)
    monkeypatch.setattr(cli, "_launchctl", lambda *_args: None)
    monkeypatch.setattr(cli, "_service_plist_path", lambda: tmp_path / "service.plist")
    monkeypatch.delenv("MLXH_MODELS_DIR", raising=False)
    cli._service_install()
    data = plistlib.loads((tmp_path / "service.plist").read_bytes())
    assert data["ProgramArguments"] == ["/opt/bin/mlxh", "serve", "--host", "127.0.0.1"]


def test_service_rejects_ephemeral_models_dir(monkeypatch):
    monkeypatch.setenv("MLXH_MODELS_DIR", "/temporary/models")
    with pytest.raises(SystemExit):
        cli._service_install()


def test_service_loaded_only_forgives_missing_service(monkeypatch):
    missing = subprocess.CompletedProcess(
        [], 113, "", "Could not find service in domain for user gui: 501"
    )
    monkeypatch.setattr(cli, "_launchctl_result", lambda *_args: missing)
    assert cli._service_loaded() is False

    real_error = subprocess.CompletedProcess([], 1, "", "Operation not permitted")
    monkeypatch.setattr(cli, "_launchctl_result", lambda *_args: real_error)
    with pytest.raises(SystemExit):
        cli._service_loaded()


def test_service_dry_run_has_no_mutation(service_env, monkeypatch, capsys):
    _home, plist = service_env
    plist.parent.mkdir(parents=True)
    plist.write_text("existing")
    monkeypatch.setattr(cli, "_service_loaded", lambda: True)
    monkeypatch.setattr(
        cli, "_write_service_plist",
        lambda *_args: pytest.fail("dry-run wrote a plist"),
    )
    monkeypatch.setattr(
        cli, "_launchctl", lambda *_args: pytest.fail("dry-run changed launchd")
    )
    cli._service_install(dry_run=True)
    out = capsys.readouterr().out
    assert "<plist" in out
    assert f"launchctl bootout {cli._service_target()}" in out
    assert f"launchctl bootstrap gui/{os.getuid()} {plist}" in out
    assert plist.read_text() == "existing"


def test_service_prefers_exported_launcher(service_env, monkeypatch, capsys):
    _home, _plist = service_env
    launcher = "/custom/bin/mlxh"
    monkeypatch.setenv("MLXH_LAUNCHER", launcher)
    monkeypatch.setattr(
        cli.shutil, "which", lambda _name: pytest.fail("PATH lookup was used")
    )
    monkeypatch.setattr(cli, "_service_loaded", lambda: False)
    cli._service_install(dry_run=True)
    data = plistlib.loads(capsys.readouterr().out.split("launchctl", 1)[0].encode())
    assert data["ProgramArguments"][0] == launcher


def test_service_initial_install(service_env, monkeypatch):
    home, plist = service_env
    calls = []
    monkeypatch.setattr(cli, "_service_loaded", lambda: False)
    monkeypatch.setattr(cli, "_port_in_use", lambda _port: False)
    monkeypatch.setattr(cli, "_launchctl", lambda *args: calls.append(args))
    cli._service_install()
    data = plistlib.loads(plist.read_bytes())
    assert data["EnvironmentVariables"]["MLXH_HOME"] == str(home.resolve())
    assert data["ProgramArguments"] == ["/opt/bin/mlxh", "serve", "--host", "127.0.0.1"]
    assert plist.stat().st_mode & 0o777 == 0o644
    assert calls == [("bootstrap", f"gui/{os.getuid()}", str(plist))]


def test_service_reinstall_orders_bootout_write_bootstrap(service_env, monkeypatch):
    _home, plist = service_env
    plist.parent.mkdir(parents=True)
    plist.write_text("old plist")
    events = []
    monkeypatch.setattr(cli, "_service_loaded", lambda: True)
    monkeypatch.setattr(cli, "_wait_for_port_release", lambda _port: True)
    monkeypatch.setattr(cli, "_launchctl",
                        lambda *args: events.append((args[0], args[1:])))
    real_write = cli._write_service_plist

    def write(path, contents):
        events.append(("write", (str(path),)))
        real_write(path, contents)

    monkeypatch.setattr(cli, "_write_service_plist", write)
    cli._service_install()
    assert [event[0] for event in events] == ["bootout", "write", "bootstrap"]
    assert plist.read_text() != "old plist"


def test_service_refuses_occupied_port_without_mutation(service_env, monkeypatch):
    _home, plist = service_env
    monkeypatch.setattr(cli, "_service_loaded", lambda: False)
    monkeypatch.setattr(cli, "_port_in_use", lambda _port: True)
    with pytest.raises(SystemExit):
        cli._service_install()
    assert not plist.exists()


def test_reinstall_keeps_old_plist_when_port_stays_busy(service_env, monkeypatch):
    _home, plist = service_env
    plist.parent.mkdir(parents=True)
    plist.write_text("old plist")
    monkeypatch.setattr(cli, "_service_loaded", lambda: True)
    monkeypatch.setattr(cli, "_launchctl", lambda *_args: None)
    monkeypatch.setattr(cli, "_wait_for_port_release", lambda _port: False)
    with pytest.raises(SystemExit):
        cli._service_install()
    assert plist.read_text() == "old plist"


def test_service_uninstall_is_idempotent_when_absent(service_env, monkeypatch):
    _home, plist = service_env
    monkeypatch.setattr(cli, "_service_loaded", lambda: False)
    cli._service_uninstall()
    assert not plist.exists()


def test_service_bootout_failure_preserves_plist(service_env, monkeypatch):
    _home, plist = service_env
    plist.parent.mkdir(parents=True)
    plist.write_text("keep me")
    monkeypatch.setattr(cli, "_service_loaded", lambda: True)

    def fail(*_args):
        raise SystemExit(1)

    monkeypatch.setattr(cli, "_launchctl", fail)
    with pytest.raises(SystemExit):
        cli._service_uninstall()
    assert plist.read_text() == "keep me"


def test_service_restart_gracefully_reloads_manager(monkeypatch, tmp_path):
    calls = []
    plist = tmp_path / "service.plist"
    plist.touch()
    monkeypatch.setattr(cli, "_service_plist_path", lambda: plist)
    monkeypatch.setattr(cli, "_service_loaded", lambda: True)
    monkeypatch.setattr(cli, "_wait_for_port_release", lambda _port: True)
    monkeypatch.setattr(cli, "_launchctl", lambda *args: calls.append(args))
    cli._service_restart()
    assert calls == [
        ("bootout", cli._service_target()),
        ("bootstrap", f"gui/{os.getuid()}", str(plist)),
    ]


def test_full_uninstall_stops_service_before_removing_home(
    service_env, monkeypatch
):
    home, plist = service_env
    plist.parent.mkdir(parents=True)
    plist.write_text("installed")
    events = []
    monkeypatch.delenv("MLXH_LAUNCHER", raising=False)
    monkeypatch.setattr(cli, "_service_uninstall",
                        lambda: events.append("service"))
    monkeypatch.setattr(cli.shutil, "rmtree",
                        lambda path, ignore_errors: events.append(("rmtree", path)))
    cli.cmd_uninstall(NS(yes=True))
    assert events == ["service", ("rmtree", home)]
