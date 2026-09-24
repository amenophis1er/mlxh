"""Install a pinned optional runtime with atomic activation and code identity checks."""

import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from .image_models import MFLUX_VERSION

REPAIR = "image support is missing or out of sync; run: mlxh images install"


def code_identity():
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return {"mlxh": importlib.metadata.version("mlxh"), "code": digest.hexdigest(),
            "mflux": MFLUX_VERSION}


def runtime_python(home):
    # An explicit mlxh[images] install can use its own interpreter.
    try:
        if importlib.metadata.version("mflux") == MFLUX_VERSION:
            return sys.executable
    except importlib.metadata.PackageNotFoundError:
        pass
    python = Path(home) / "images" / "current" / "bin" / "python"
    try:
        result = subprocess.run(
            [str(python), "-I", "-c",
             "import json, importlib.metadata as m; from mlxh.image_runtime import code_identity; "
             "d=code_identity(); d['mflux']=m.version('mflux'); print(json.dumps(d))"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        if json.loads(result.stdout) == code_identity():
            # Resolve the environment directory, not the python symlink (which
            # points to the system interpreter). Running servers keep a stable
            # sys.path even when a later install switches the current symlink.
            return str(python.parent.parent.resolve() / "bin" / "python")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    raise RuntimeError(REPAIR)


def _wheel(destination):
    """Snapshot the calling package, including editable checkouts, into a wheel.

    The private runtime must run exactly this code, including unreleased builds.
    mlxh is pure Python; copy its distribution metadata and regenerate RECORD.
    """
    dist = importlib.metadata.distribution("mlxh")
    version = dist.version
    metadata = f"mlxh-{version}.dist-info"
    wheel = destination / f"mlxh-{version}-py3-none-any.whl"
    package = Path(__file__).parent
    files = {f"mlxh/{p.relative_to(package).as_posix()}": p.read_bytes()
             for p in sorted(package.rglob("*.py"))}
    for name in ("METADATA", "WHEEL", "entry_points.txt"):
        content = dist.read_text(name)
        if content:
            files[f"{metadata}/{name}"] = content.encode()
    record = io.StringIO()
    writer = csv.writer(record)
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
            sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow((name, "sha256=" + sha, len(data)))
        writer.writerow((f"{metadata}/RECORD", "", ""))
        archive.writestr(f"{metadata}/RECORD", record.getvalue())
    return wheel


def install(home):
    """Stage beside the active environment; a failed install never changes it."""
    import fcntl

    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to install image support")
    root = Path(home) / "images"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            runtime_python(home)
            return
        except RuntimeError:
            pass
        current_link = root / "current"
        previous = current_link.resolve() if current_link.exists() else None
        stage = Path(tempfile.mkdtemp(prefix="env-", dir=root))
        log_path = root / "install.log"
        with log_path.open("w") as log:
            try:
                wheel = _wheel(stage)
                python = stage / "bin" / "python"
                subprocess.run([uv, "venv", "--allow-existing", "--python", sys.executable, str(stage)],
                               stdout=log, stderr=log, check=True)
                subprocess.run([uv, "pip", "install", "--python", str(python),
                                str(wheel), f"mflux=={MFLUX_VERSION}"],
                               stdout=log, stderr=log, check=True)
                probe = subprocess.run(
                    [str(python), "-I", "-c",
                     "import json; from mlxh.image_runtime import code_identity; "
                     "from mflux.models.flux.variants.txt2img.flux import Flux1; "
                     "from mflux.models.flux2.variants import Flux2Klein; "
                     "from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage; "
                     "print(json.dumps(code_identity()))"],
                    stdout=subprocess.PIPE, stderr=log, text=True, check=True,
                )
                if json.loads(probe.stdout) != code_identity():
                    raise RuntimeError("installed image runtime identity differs")
                (stage / "mlxh-runtime.json").write_text(json.dumps(code_identity(), indent=2))
                link = root / "next"
                link.unlink(missing_ok=True)
                link.symlink_to(stage.name, target_is_directory=True)
                os.replace(link, root / "current")
                # Keep the activated runtime and one rollback candidate. Remove
                # older staged environments only after activation succeeded.
                environments = sorted(
                    (p for p in root.glob("env-*") if p.is_dir()),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                keep = {stage.resolve()}
                if previous and previous != stage.resolve() and previous.exists():
                    keep.add(previous)
                for environment in environments:
                    if environment.resolve() not in keep:
                        shutil.rmtree(environment)
            except Exception as exc:
                # Keep failed staging/logs for diagnosis; never remove the active runtime.
                raise RuntimeError(f"image runtime installation failed; see {log_path}") from exc
