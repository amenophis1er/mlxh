import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from mlxh import image_runtime as runtime


def test_wheel_contains_current_source(tmp_path):
    wheel = runtime._wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read("mlxh/image_runtime.py") == Path(runtime.__file__).read_bytes()
        assert any(name.endswith("/RECORD") for name in archive.namelist())


def test_mismatched_private_runtime_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "old")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: NS(stdout=json.dumps({"mlxh": "old"})))
    with pytest.raises(RuntimeError, match="mlxh images install"):
        runtime.runtime_python(tmp_path)


def test_runtime_path_keeps_venv_interpreter(tmp_path, monkeypatch):
    identity = {"mlxh": "1", "code": "abc", "mflux": runtime.MFLUX_VERSION}
    monkeypatch.setattr(runtime, "code_identity", lambda: identity)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "old")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: NS(stdout=json.dumps(identity)))
    root = tmp_path / "images"
    (root / "env-one/bin").mkdir(parents=True)
    (root / "current").symlink_to("env-one")
    (root / "env-one/bin/python").symlink_to("/usr/bin/python3")
    result = runtime.runtime_python(tmp_path)
    assert result == str(root / "env-one/bin/python")


@pytest.mark.parametrize("fail", [False, True])
def test_staged_install_activates_only_after_validation(tmp_path, monkeypatch, fail):
    root = tmp_path / "images"
    root.mkdir()
    old = root / "old"
    old.mkdir()
    (root / "current").symlink_to("old")
    stale = root / "env-stale"
    stale.mkdir()
    monkeypatch.setattr(runtime, "runtime_python", lambda _: (_ for _ in ()).throw(RuntimeError("stale")))
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/bin/uv")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "pip" in argv and fail:
            raise subprocess.CalledProcessError(1, argv)
        return NS(stdout=json.dumps(runtime.code_identity()))

    monkeypatch.setattr(runtime.subprocess, "run", run)
    if fail:
        with pytest.raises(RuntimeError, match="install.log"):
            runtime.install(tmp_path)
        assert (root / "current").resolve() == old
    else:
        runtime.install(tmp_path)
        assert any("Flux2KleinEdit" in str(arg) for call in calls for arg in call)
        assert (root / "current").resolve() != old
        assert json.loads((root / "current/mlxh-runtime.json").read_text()) == runtime.code_identity()
        assert not stale.exists()
        assert old.exists()  # one previous environment remains available for rollback
    assert old.is_dir()
    assert len(calls) >= 2


def test_install_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "runtime_python", lambda _: "/managed/python")
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/bin/uv")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: pytest.fail("reinstalled"))
    runtime.install(tmp_path)
