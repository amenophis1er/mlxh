"""Install a pinned optional runtime with atomic activation and code identity checks.

Two runtimes use it: image generation (mflux), and Inkling 2 (an mlx-lm that
has its model, which mlxh's own mlx-lm does not: newer mlx-lm moved the models
the Bonsai packs import, so the main environment cannot simply move up).
Each lives in MLXH_HOME/<name>/current, with its own copy of this mlxh.
"""

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

from dataclasses import dataclass

from .image_models import MFLUX_VERSION

REPAIR = "image support is missing or out of sync; run: mlxh images install"

# ml-explore/mlx-lm#2086 ("Add Inkling 2 Mini model"), reviewed at this commit.
# Drop this runtime for the main environment once a released mlx-lm has it and
# the Bonsai packs load on that release.
INKLING_MLX_LM = "e2b5bbd3bd9d186dea895e60802aa765e70b9d93"
_INKLING_REQ = f"mlx-lm @ git+https://github.com/ml-explore/mlx-lm@{INKLING_MLX_LM}"


@dataclass(frozen=True)
class Runtime:
    name: str          # directory under MLXH_HOME, and the CLI word: mlxh <name> install
    key: str           # what the identity pins, beside mlxh's own code
    version: str
    requirements: tuple
    overrides: tuple   # replace mlxh's own pins in that environment
    installed: str     # Python expression, in the runtime (m = importlib.metadata): what `key` is there
    probe: str         # imports that must work there
    repair: str

    def here(self):
        """This interpreter already has it (an explicit install of the extra)."""
        try:
            return eval(self.installed, {"m": importlib.metadata, "json": json}) == self.version
        except Exception:
            return False


IMAGES = Runtime(
    "images", "mflux", MFLUX_VERSION, (f"mflux=={MFLUX_VERSION}",), (),
    "m.version('mflux')",
    "from mflux.models.flux.variants.txt2img.flux import Flux1; "
    "from mflux.models.flux2.variants import Flux2Klein; "
    "from mflux.models.flux2.variants import Flux2KleinEdit; "
    "from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage",
    REPAIR,
)
INKLING = Runtime(
    "inkling", "mlx-lm", INKLING_MLX_LM, (_INKLING_REQ,), (_INKLING_REQ,),
    "json.loads(m.distribution('mlx-lm').read_text('direct_url.json'))['vcs_info']['commit_id']",
    "import mlx_lm.models.inkling2",
    "Inkling 2 support is missing or out of sync; run: mlxh inkling install",
)


def code_identity(runtime=IMAGES):
    digest = hashlib.sha256()
    package = Path(__file__).parent
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return {"mlxh": importlib.metadata.version("mlxh"), "code": digest.hexdigest(),
            runtime.key: runtime.version}


def _identity_script(runtime):
    return ("import json, importlib.metadata as m; from mlxh.image_runtime import code_identity, "
            f"{runtime.name.upper()} as r; d=code_identity(r); d[r.key]=({runtime.installed}); "
            "print(json.dumps(d))")


def runtime_python(home, runtime=IMAGES):
    # An explicit install of the extra can use its own interpreter.
    if runtime.here():
        return sys.executable
    python = Path(home) / runtime.name / "current" / "bin" / "python"
    try:
        result = subprocess.run(
            [str(python), "-I", "-c", _identity_script(runtime)],
            capture_output=True, text=True, timeout=10, check=True,
        )
        if json.loads(result.stdout) == code_identity(runtime):
            # Resolve the environment directory, not the python symlink (which
            # points to the system interpreter). Running servers keep a stable
            # sys.path even when a later install switches the current symlink.
            return str(python.parent.parent.resolve() / "bin" / "python")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    raise RuntimeError(runtime.repair)


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


def install(home, runtime=IMAGES):
    """Stage beside the active environment; a failed install never changes it."""
    import fcntl

    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError(f"uv is required to install {runtime.name} support")
    root = Path(home) / runtime.name
    root.mkdir(parents=True, exist_ok=True)
    with (root / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            runtime_python(home, runtime)
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
                pins = []
                if runtime.overrides:
                    overrides = stage / "overrides.txt"
                    overrides.write_text("\n".join(runtime.overrides) + "\n")
                    pins = ["--overrides", str(overrides)]
                subprocess.run([uv, "pip", "install", "--python", str(python), *pins,
                                str(wheel), *runtime.requirements],
                               stdout=log, stderr=log, check=True)
                probe = subprocess.run(
                    [str(python), "-I", "-c", f"{runtime.probe}; {_identity_script(runtime)}"],
                    stdout=subprocess.PIPE, stderr=log, text=True, check=True,
                )
                if json.loads(probe.stdout) != code_identity(runtime):
                    raise RuntimeError(f"installed {runtime.name} runtime identity differs")
                (stage / "mlxh-runtime.json").write_text(json.dumps(code_identity(runtime), indent=2))
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
                # The log is what matters for diagnosis; a failed staging env is
                # ~1.3 GB of nothing, so drop it. Never touch the active runtime.
                if stage.resolve() != (previous or stage.resolve()):
                    shutil.rmtree(stage, ignore_errors=True)
                raise RuntimeError(f"{runtime.name} runtime installation failed; see {log_path}") from exc
