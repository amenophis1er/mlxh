"""mlxh — a small harness for running local MLX models with an OpenAI API.

Commands:
  mlxh run <repo-or-name>             chat now, pulling first if needed
  mlxh pull <hf-repo> [--name NAME]   download a model from Hugging Face
  mlxh link <path> [--name NAME]      symlink an existing local model dir in
  mlxh list                           show available models
  mlxh mv <name> <new-name>           rename a model
  mlxh rm <name>                      remove a model (links: symlink only)
  mlxh serve <name> [--port N ...]    OpenAI + Anthropic compatible API server
  mlxh images install                install optional local image generation
  mlxh image [model] [prompt...]     generate images interactively or once
  mlxh status [--json]                show live stats of the local server
  mlxh service <action>               manage the login LaunchAgent
  mlxh launch <agent> [--model NAME]  run a coding agent (claude, codex, pi)
  mlxh chat <name> [chat args...]     terminal chat (tools, images, streaming)
  mlxh config [key [value]]           show or set config
  mlxh uninstall                      remove mlxh and everything it manages

Models live in ONE directory and the filesystem is the registry: every
subdirectory of the models dir that holds a config.json is a usable model.
`pull` downloads there; `link` drops a symlink there. The location is the
`models_dir` config key, overridable with $MLXH_MODELS_DIR.

Config keys (mlxh config <key> <value>):
  port, host          server defaults (port 1060 — "MLX" in Roman numerals)
  models_dir          where models live (default $MLXH_HOME/models)
  max_queued          pending generations beyond the active one before 503 (4)
  max_tokens_cap      server-side ceiling on max_tokens, 0 = unlimited (16384)
  memory_limit_gb     MLX memory limit, 0 = auto (80% RAM), -1 = off
  cache_limit_gb      MLX buffer-cache limit, 0 = off
  gen_timeout_s       hard stop for one generation, 0 = off (600)
  max_prompt_tokens   reject prompts bigger than this, 0 = off (8192)
  prompt_cache        reuse KV blocks across requests (true)
  thinking            model reasoning: auto / on / off (auto)
  chat_tools          load ~/.mlxh/tools.py in chat by default (false)
  service_model       model loaded by `mlxh service install` (unset)
  max_image_pixels    image width * height ceiling (4194304)
  image_steps         denoising steps, 0 = model default (0)

State lives under $MLXH_HOME (default ~/.mlxh): venv, app code, config,
models, HF cache. Uninstall removes exactly that plus the launcher.
"""

import argparse
import json
import os
import plistlib
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import ui
from .image_models import (
    image_metadata, model_kind, IMAGE_REPO, IMAGE_REVISION, IMAGE_CATALOG,
)

HOME = Path(os.environ.get("MLXH_HOME", Path.home() / ".mlxh"))
CONFIG = HOME / "config.json"
DEFAULTS = {
    "port": 1060,
    "host": "127.0.0.1",
    "models_dir": str(HOME / "models"),
    "max_queued": 4,
    "max_tokens_cap": 16384,
    "memory_limit_gb": 0.0,
    "cache_limit_gb": 0.0,
    "gen_timeout_s": 600,
    "max_prompt_tokens": 8192,
    "prompt_cache": True,
    "thinking": "auto",
    "chat_tools": False,
    "service_model": "",
    "max_image_pixels": 4194304,
    "image_steps": 0,
}
def _bool(v):
    if v.lower() in ("1", "true", "on", "yes"):
        return True
    if v.lower() in ("0", "false", "off", "no"):
        return False
    raise ValueError(v)


KEY_TYPES = {
    "port": int, "host": str, "models_dir": str, "max_queued": int,
    "max_tokens_cap": int, "memory_limit_gb": float, "cache_limit_gb": float,
    "gen_timeout_s": int, "max_prompt_tokens": int, "prompt_cache": _bool,
    "thinking": str, "chat_tools": _bool, "service_model": str,
    "max_image_pixels": int, "image_steps": int,
}

SERVICE_LABEL = "com.mlxh.serve"


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG.exists():
        stored = json.loads(CONFIG.read_text())
        stored.pop("models", None)  # registry from pre-0.2 layouts; now unused
        cfg.update(stored)
    return cfg


def save_config(cfg):
    HOME.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2))


def models_dir(cfg):
    return Path(os.environ.get("MLXH_MODELS_DIR") or cfg["models_dir"]).expanduser()


def is_model(path):
    try:
        return image_metadata(path) is not None or (path / "config.json").is_file()
    except ValueError:
        # Keep malformed image packs discoverable so commands can explain the
        # metadata error instead of making the model silently disappear.
        return (path / ".mlxh.json").is_file() or (path / "config.json").is_file()


def checked_model_kind(path):
    try:
        return model_kind(path)
    except ValueError as exc:
        ui.fail("unsupported model metadata", str(exc))


def model_supports_images(path):
    """Read enough local model metadata to advertise image input safely."""
    try:
        cfg = json.loads((Path(path) / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(cfg, dict):
        return False
    if str(cfg.get("model_type", "")).startswith("prism_"):
        return bool((cfg.get("components") or {}).get("vision"))
    return any(cfg.get(key) is not None for key in (
        "vision_config", "vision_tower", "mm_vision_tower",
        "multimodal_projector_config",
    ))


def discover(cfg):
    root = models_dir(cfg)
    if not root.is_dir():
        return {}
    return {p.name: p for p in sorted(root.iterdir()) if p.is_dir() and is_model(p)}


def source_of(path):
    """Where a model came from: repo@rev if mlxh pulled it, best effort otherwise."""
    meta = path / ".mlxh.json"
    if meta.is_file():
        try:
            m = json.loads(meta.read_text())
            return f"{m['repo']}@{m.get('revision', '')[:7]}".rstrip("@")
        except (json.JSONDecodeError, KeyError):
            pass
    # Dir downloaded by other HF tooling: local metadata has the revision only.
    for f in (path / ".cache" / "huggingface" / "download").glob("*.metadata"):
        try:
            rev = f.read_text().splitlines()[0].strip()
            if len(rev) == 40:
                return f"hf@{rev[:7]}"
        except (OSError, IndexError):
            continue
    return "-"


def resolve(cfg, name):
    path = models_dir(cfg) / name
    if not is_model(path):
        names = ", ".join(discover(cfg)) or "(none)"
        ui.fail(f"no model named '{name}'",
                f"models dir: {models_dir(cfg)}",
                f"available:  {names}",
                hint="mlxh pull <hf-repo> downloads one; mlxh search <query> finds them")
    return str(path)


def pick_model(cfg, purpose):
    """Choose an installed model when none was named."""
    models = list(discover(cfg))
    if not models:
        ui.fail("no models installed",
                hint="mlxh search <query> finds MLX models; mlxh pull <repo-id> installs one")
    if len(models) == 1:
        ui.note(f"using {models[0]}")
        return models[0]
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        ui.fail(f"which model do you want to {purpose}?",
                f"available: {', '.join(models)}")
    paths = discover(cfg)
    idx = ui.select(f"select a model to {purpose}  (↑/↓, enter)",
                    models, [source_of(paths[n]) for n in models])
    return models[idx]


def cmd_search(args):
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi

    api = HfApi()
    results = list(api.list_models(
        search=args.query, filter="mlx", sort="downloads", limit=args.limit,
        expand=["downloads", "lastModified", "safetensors"],
    ))
    if not results:
        print(f"no MLX models match '{args.query}'")
        ui.note("try https://huggingface.co/models?library=mlx")
        return

    def size_of(repo_id):
        # download size = sum of the repo's files; one small extra call per row
        try:
            info = api.model_info(repo_id, files_metadata=True)
            total = sum(s.size or 0 for s in info.siblings)
            return f"{total / 1e9:.1f} GB" if total else "-"
        except Exception:
            return "-"

    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(size_of, [m.id for m in results]))

    def params_of(m):
        st = getattr(m, "safetensors", None)
        if not st or not st.total:
            return "-"
        n = st.total
        return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"

    rw = max(len("REPO"), max(len(m.id) for m in results))
    print(ui.dim(f"{'REPO':{rw}} {'PARAMS':>7} {'SIZE':>8} {'DOWNLOADS':>10}  UPDATED"))
    for m, size in zip(results, sizes):
        updated = m.last_modified.strftime("%Y-%m-%d") if m.last_modified else "-"
        print(f"{m.id:{rw}} {params_of(m):>7} {size:>8} {m.downloads or 0:>10,}  {updated}")
    print()
    ui.note("install one with: mlxh pull <repo-id>")


def total_ram_bytes():
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        try:
            import subprocess
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        except Exception:
            return 0


def do_pull(cfg, repo, name, force=False, kind="auto", backend=None):
    image_spec = IMAGE_CATALOG.get(repo)
    image = image_spec is not None
    if backend and (not image or backend != image_spec["backend"]):
        ui.fail(f"backend '{backend}' is not supported for repository '{repo}'",
                hint="supported image repositories: " + ", ".join(IMAGE_CATALOG))
    if kind == "image" and not image:
        ui.fail("unsupported image repository",
                hint="supported image repositories: " + ", ".join(IMAGE_CATALOG))
    if kind == "language" and image:
        ui.fail("this repository contains an image generation model")
    dest = models_dir(cfg) / name
    if dest.exists():
        ui.fail(f"'{name}' already exists",
                f"at {dest}",
                hint=f"pick another with --name, or `mlxh rm {name}` first")
    if image:
        _ensure_image_runtime(interactive=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HOME / "hf-cache"))
    from datetime import datetime, timezone
    from huggingface_hub import HfApi, snapshot_download

    fixed_revision = IMAGE_REVISION if repo == IMAGE_REPO else None
    repo_info = None
    try:
        repo_info = HfApi().model_info(repo, files_metadata=True,
                                      **({"revision": fixed_revision} if fixed_revision else {}))
        if not image and (getattr(repo_info, "pipeline_tag", None) == "text-to-image"
                          or getattr(repo_info, "library_name", None) == "diffusers"):
            ui.fail("unsupported image repository", hint=f"verified model: {IMAGE_REPO}")
        size = sum(s.size or 0 for s in repo_info.siblings)
    except Exception as exc:
        if image:
            ui.fail("could not inspect image repository", f"{type(exc).__name__}: {exc}",
                    hint="check Hugging Face access and try again")
        size = 0  # can't size it (offline, gated, ...): proceed without guardrails
    if size:
        free = shutil.disk_usage(dest.parent).free
        if size + 2e9 > free:  # keep a 2 GB margin
            ui.fail(f"{repo} does not fit on disk",
                    f"download size  {size / 1e9:8.1f} GB",
                    f"free space     {free / 1e9:8.1f} GB  ({dest.parent})")
        ram = total_ram_bytes()
        if ram and size > 0.9 * ram and not force:
            ui.fail(f"{repo} won't load on this machine",
                    f"model weights  {size / 1e9:8.1f} GB",
                    f"unified memory {ram / 1e9:8.0f} GB",
                    hint="--force downloads anyway (e.g. for another machine)")

    ui.step(f"downloading {repo}{f' ({size / 1e9:.1f} GB)' if size else ''}")
    revision = fixed_revision or getattr(repo_info, "sha", None) or ""
    try:
        snapshot_download(repo, local_dir=str(dest), **({"revision": revision} if image else {}))
    except Exception as e:
        shutil.rmtree(dest, ignore_errors=True)
        ui.fail("download failed",
                f"{type(e).__name__}: {e}",
                hint="check the repo id with `mlxh search`; gated repos need `hf auth login`")
    if not image:
        try:
            revision = HfApi().model_info(repo).sha or ""
        except Exception:
            revision = ""
    (dest / ".mlxh.json").write_text(json.dumps({
        "repo": repo,
        "revision": revision,
        "pulled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **({"kind": "image", **image_spec} if image else {}),
    }, indent=2))
    ui.ok(f"pulled {repo}@{revision[:7]} as '{name}'")


def cmd_pull(args):
    cfg = load_config()
    name = args.name or args.repo.split("/")[-1]
    do_pull(cfg, args.repo, name, force=args.force,
            kind=getattr(args, "kind", "auto"), backend=getattr(args, "backend", None))
    ui.note(f"serve: mlxh serve {name}" if model_kind(models_dir(cfg) / name) == "image"
            else f"chat: mlxh chat {name}    serve: mlxh serve {name}")


def _ensure_image_runtime(interactive=False):
    from .image_runtime import runtime_python, install
    try:
        return runtime_python(HOME)
    except RuntimeError:
        if interactive and sys.stdin.isatty() and sys.stdout.isatty():
            ui.note("Image support needs a pinned runtime (several hundred MB plus dependencies).")
            if input("Install image support now? [y/N] ").strip().lower() in ("y", "yes"):
                try:
                    install(HOME)
                    return runtime_python(HOME)
                except RuntimeError as exc:
                    ui.fail(str(exc))
        ui.fail("install image support with: mlxh images install")


def cmd_images(_args):
    from .image_runtime import install
    try:
        install(HOME)
    except RuntimeError as exc:
        ui.fail(str(exc))
    ui.ok("image support is installed")


def _require_language(path):
    if checked_model_kind(path) == "image":
        ui.fail("this model generates images and does not support chat",
                hint="use mlxh serve and /v1/images/generations")


def repo_installed_as(cfg, repo):
    """Name of an installed model whose recorded source repo matches, if any."""
    for name, path in discover(cfg).items():
        meta = path / ".mlxh.json"
        if meta.is_file():
            try:
                if json.loads(meta.read_text()).get("repo") == repo:
                    return name
            except json.JSONDecodeError:
                pass
    return None


def cmd_run(args):
    cfg = load_config()
    name = args.target or pick_model(cfg, "run")
    if "/" in name:  # a Hugging Face repo id
        installed = repo_installed_as(cfg, name)
        if installed:
            name = installed
        else:
            name = name.split("/")[-1]
            if not is_model(models_dir(cfg) / name):
                do_pull(cfg, args.target, name, force=args.force, kind="language")
    path = resolve(cfg, name)
    _require_language(path)
    os.execv(sys.executable, [
        sys.executable, "-m", "mlxh.chat_cli", "--model-path", path,
        *_chat_args(cfg, args.rest),
    ])


def cmd_link(args):
    cfg = load_config()
    target = Path(args.path).expanduser().resolve()
    if not is_model(target):
        sys.exit(f"{target} does not look like a model directory (no config.json)")
    name = args.name or target.name
    dest = models_dir(cfg) / name
    if dest.exists() or dest.is_symlink():
        ui.fail(f"'{name}' already exists", f"at {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(target)
    ui.ok(f"linked '{name}' -> {target}")
    ui.note(f"mlxh rm {name} removes only the link, never the files")


def cmd_list(_args):
    cfg = load_config()
    models = discover(cfg)
    if not models:
        print(f"no models in {models_dir(cfg)}")
        ui.note("mlxh search <query> finds MLX models; mlxh pull <repo-id> installs one")
        return
    rows = []
    for name, path in models.items():
        n = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        rows.append((name, "linked" if path.is_symlink() else "pulled",
                     f"{n / 1e9:.1f} GB", source_of(path),
                     str(path.resolve()) if path.is_symlink() else ""))
    nw = max(len("NAME"), max(len(r[0]) for r in rows))
    sw = max(len("SOURCE"), max(len(r[3]) for r in rows))
    cols = shutil.get_terminal_size().columns if sys.stdout.isatty() else 10**9
    print(ui.dim(f"{'NAME':{nw}}  {'INSTALL':6} {'KIND':12} {'SIZE':>8}  SOURCE"))
    home = str(Path.home())
    for name, kind, size, source, target in rows:
        try:
            meta = image_metadata(models[name])
            quantization = meta.get("quantization_bits") if meta else None
            precision = f"{quantization}bit" if quantization else "full"
            family = f"image/{precision}" if meta else "language"
        except ValueError as exc:
            family = "unsupported"
            ui.note(f"{name}: unsupported model metadata: {exc}")
        line = f"{name:{nw}}  {kind:6} {family:12} {size:>8}  {source:{sw}}"
        if target:
            if target.startswith(home):
                target = "~" + target[len(home):]
            if len(line) + len(target) + 5 <= cols:
                line += f"  -> {target}"
            else:
                # keep the row intact; the link target gets its own line
                print(line.rstrip())
                print(ui.dim(f"{'':{nw}}  -> {target}"))
                continue
        print(line.rstrip())


def cmd_mv(args):
    cfg = load_config()
    root = models_dir(cfg)
    old, new = root / args.name, root / args.new_name
    if not is_model(old):
        ui.fail(f"no model named '{args.name}'", f"models dir: {root}")
    if "/" in args.new_name or not args.new_name.strip():
        ui.fail(f"invalid name '{args.new_name}'")
    if new.exists() or new.is_symlink():
        ui.fail(f"'{args.new_name}' already exists", f"at {new}")
    old.rename(new)
    ui.ok(f"renamed '{args.name}' -> '{args.new_name}'")
    ui.note(f"chat: mlxh chat {args.new_name}    serve: mlxh serve {args.new_name}")


def cmd_rm(args):
    cfg = load_config()
    path = models_dir(cfg) / args.name
    if path.is_symlink():
        target = path.resolve()
        path.unlink()
        ui.ok(f"removed link '{args.name}'")
        ui.note(f"files at {target} untouched")
    elif is_model(path):
        shutil.rmtree(path)
        ui.ok(f"deleted {path}")
    else:
        ui.fail(f"no model named '{args.name}'", f"models dir: {models_dir(cfg)}")


def serve_argv(cfg, name, path, overrides=None):
    o = overrides or {}

    def pick(key):
        return o.get(key) if o.get(key) is not None else cfg[key]

    python = _ensure_image_runtime() if checked_model_kind(path) == "image" else sys.executable
    return [
        python, *(["-I"] if python != sys.executable else []), "-m", "mlxh.serve_app",
        "--model-path", path, "--name", name,
        "--port", str(pick("port")),
        "--host", str(pick("host")),
        "--max-queued", str(pick("max_queued")),
        "--max-tokens-cap", str(pick("max_tokens_cap")),
        "--memory-limit-gb", str(pick("memory_limit_gb")),
        "--cache-limit-gb", str(pick("cache_limit_gb")),
        "--gen-timeout-s", str(pick("gen_timeout_s")),
        "--max-prompt-tokens", str(pick("max_prompt_tokens")),
        "--prompt-cache", str(pick("prompt_cache")),
        "--thinking", str(pick("thinking")),
        "--max-image-pixels", str(pick("max_image_pixels")),
        "--image-steps", str(pick("image_steps")),
    ]


def cmd_serve(args):
    cfg = load_config()
    name = args.name or pick_model(cfg, "serve")
    path = resolve(cfg, name)
    overrides = {k: getattr(args, k) for k in
                 ("port", "host", "max_queued", "max_tokens_cap",
                  "memory_limit_gb", "cache_limit_gb", "gen_timeout_s",
                  "max_prompt_tokens", "prompt_cache", "thinking",
                  "max_image_pixels", "image_steps")}
    argv = serve_argv(cfg, name, path, overrides)
    os.execv(argv[0], argv)


def _fetch_info(port):
    """Fetch one diagnostics snapshot from the local server."""
    import urllib.request

    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/mlxh/info", timeout=2
    ) as response:
        data = json.loads(response.read())
    if not isinstance(data, dict):
        raise ValueError("/mlxh/info did not return an object")
    return data


def _human_value(value, formatter=str):
    return "—" if value is None else formatter(value)


def _format_uptime(value):
    value = int(value)
    if value < 60:
        return f"{value}s"
    if value < 3600:
        return f"{value // 60}m"
    return f"{value // 3600}h"


def _format_tokens(value):
    value = int(value)
    if value < 1000:
        return str(value)
    scaled = value / 1000
    return f"{scaled:.1f}k" if scaled < 100 else f"{scaled:.0f}k"


def _format_gb(value):
    return f"{int(value) / 1e9:.1f} GB"


def cmd_status(args):
    port = load_config()["port"]
    try:
        info = _fetch_info(port)
    except Exception:
        print(f"no mlxh server running on 127.0.0.1:{port}", file=sys.stderr)
        raise SystemExit(1)

    if args.json:
        print(json.dumps({**info, "port": port}, indent=2))
        return

    runtime = info.get("runtime") or {}
    mlx = info.get("mlx") or {}
    ready_value = runtime.get("ready")
    if ready_value is None:
        state = "—"
    elif not ready_value:
        state = "LOADING"
    elif runtime.get("busy"):
        state = "BUSY"
    else:
        state = "IDLE"
    values = [
        _human_value(info.get("model")),
        _human_value(runtime.get("pid")),
        _human_value(runtime.get("uptime_s"), _format_uptime),
        state,
        _human_value(runtime.get("queue_depth")),
        _human_value(runtime.get("requests")),
        _human_value(runtime.get("prompt_tokens"), _format_tokens),
        _human_value(runtime.get("tokens_generated"), _format_tokens),
        _human_value(mlx.get("active_memory_bytes"), _format_gb),
        _human_value(mlx.get("cache_memory_bytes"), _format_gb),
        _human_value(mlx.get("last_peak_memory_bytes"), _format_gb),
    ]
    headers = ["MODEL", "PID", "UPTIME", "STATE", "QUEUE", "REQ",
               "PROMPT", "OUTPUT", "ACTIVE", "CACHE", "PEAK"]
    if info.get("model_kind") == "image":
        headers[6:8] = ["IMAGES"]
        values[6:8] = [_human_value(runtime.get("images_generated"))]
    widths = [max(len(header), len(str(value)))
              for header, value in zip(headers, values)]
    print(ui.dim("  ".join(f"{header:{width}}"
                           for header, width in zip(headers, widths))))
    print("  ".join(f"{value:{width}}"
                    for value, width in zip(values, widths)).rstrip())
    if runtime.get("engine_version") is None:
        ui.note("warning: this is an older mlxh server; restart it for live diagnostics")


def service_plist(mlxh_bin: str, model: str, log_path: str,
                  mlxh_home: str) -> str:
    """Return a launchd plist for the persistent localhost server."""
    data = {
        "Label": SERVICE_LABEL,
        "ProgramArguments": [mlxh_bin, "serve", model, "--host", "127.0.0.1"],
        "KeepAlive": True,
        "ThrottleInterval": 60,
        "EnvironmentVariables": {
            "PATH": f"{Path(mlxh_bin).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
            "MLXH_HOME": mlxh_home,
        },
        "StandardOutPath": log_path,
        "StandardErrorPath": log_path,
    }
    return plistlib.dumps(data, fmt=plistlib.FMT_XML).decode()


def _service_plist_path():
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_LABEL}.plist"


def _service_target():
    return f"gui/{os.getuid()}/{SERVICE_LABEL}"


def _launchctl_result(*args):
    try:
        return subprocess.run(
            ["launchctl", *args], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        ui.fail("could not run launchctl", str(exc))


def _missing_service(result):
    message = f"{result.stdout}\n{result.stderr}".lower()
    return any(text in message for text in (
        "could not find service", "service not found", "no such process"
    ))


def _service_loaded():
    result = _launchctl_result("print", _service_target())
    if result.returncode == 0:
        return True
    if _missing_service(result):
        return False
    ui.fail("could not inspect the mlxh service",
            result.stderr.strip() or result.stdout.strip())


def _launchctl(*args):
    result = _launchctl_result(*args)
    if result.returncode:
        ui.fail(f"launchctl {' '.join(args)} failed",
                result.stderr.strip() or result.stdout.strip())


def _port_in_use(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _wait_for_port_release(port, timeout=3.0):
    deadline = time.monotonic() + timeout
    while _port_in_use(port):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _write_service_plist(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "w") as file:
            file.write(contents)
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _service_install(dry_run=False):
    if "MLXH_MODELS_DIR" in os.environ:
        ui.fail("MLXH_MODELS_DIR cannot be used by the login service",
                hint="persist it with `mlxh config models_dir PATH`, then retry")
    cfg = load_config()
    model = cfg["service_model"]
    if not model:
        ui.fail("service_model is not configured",
                hint="run `mlxh config service_model MODEL` first")
    resolve(cfg, model)
    launcher = os.environ.get("MLXH_LAUNCHER") or shutil.which("mlxh")
    if not launcher:
        ui.fail("could not find the mlxh launcher on PATH")
    launcher = str(Path(launcher).expanduser().resolve())
    mlxh_home = str(HOME.expanduser().resolve())
    path = _service_plist_path()
    contents = service_plist(
        launcher, model, str(Path(mlxh_home) / "service.log"), mlxh_home
    )
    loaded = _service_loaded()

    if dry_run:
        print(contents, end="")
        if loaded:
            print(f"launchctl bootout {_service_target()}")
        print(f"launchctl bootstrap gui/{os.getuid()} {path}")
        return

    port = cfg["port"]
    if loaded:
        _launchctl("bootout", _service_target())
        if not _wait_for_port_release(port):
            ui.fail(f"port {port} is still in use after stopping the service",
                    hint="stop the process using it, then retry")
    elif _port_in_use(port):
        ui.fail(f"port {port} is already in use",
                hint="stop the existing server, then retry")

    _write_service_plist(path, contents)
    _launchctl("bootstrap", f"gui/{os.getuid()}", str(path))
    ui.ok(f"installed {SERVICE_LABEL} for model '{model}'")
    ui.note(f"logs append to {Path(mlxh_home) / 'service.log'}")


def _service_uninstall(dry_run=False):
    path = _service_plist_path()
    loaded = _service_loaded()
    if dry_run:
        if loaded:
            print(f"launchctl bootout {_service_target()}")
        print(f"rm {path}")
        return
    if loaded:
        _launchctl("bootout", _service_target())
    path.unlink(missing_ok=True)
    ui.ok(f"removed {SERVICE_LABEL}")


def _service_restart(dry_run=False):
    command = ("kickstart", "-k", _service_target())
    if dry_run:
        print(f"launchctl {' '.join(command)}")
        return
    _launchctl(*command)
    ui.ok(f"restarted {SERVICE_LABEL}")


def cmd_service(args):
    if args.action == "install":
        _service_install(args.dry_run)
    elif args.action == "uninstall":
        _service_uninstall(args.dry_run)
    else:
        _service_restart(args.dry_run)


def _chat_args(cfg, rest):
    if cfg["chat_tools"] and "--tools" not in rest and "--no-tools" not in rest:
        rest = ["--tools", *rest]
    return rest


AGENTS = {
    # agent -> (env for a server at PORT, extra argv given MODEL)
    "claude": lambda port, model: (
        {"ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
         "ANTHROPIC_AUTH_TOKEN": "mlxh", "ANTHROPIC_API_KEY": ""},
        ["--model", model]),
    # ChatGPT-account Codex ignores OPENAI_BASE_URL; a model_provider
    # override is the documented way to point it at a custom server.
    "codex": lambda port, model: (
        {"MLXH_API_KEY": "mlxh"},
        ["-c", "model_providers.mlxh.name=mlxh",
         "-c", f'model_providers.mlxh.base_url="http://127.0.0.1:{port}/v1"',
         "-c", "model_providers.mlxh.wire_api=responses",
         "-c", 'model_providers.mlxh.env_key="MLXH_API_KEY"',
         "-c", "model_provider=mlxh",
         "--model", model]),
    "pi": lambda port, model: (
        {},  # pi is configured via ~/.pi/agent/models.json, not env vars
        ["--provider", "mlxh", "--model", model, "--api-key", "mlxh"]),
}


def pi_register_provider(port, model, path=None, supports_images=False):
    """Merge an 'mlxh' provider into pi's models.json (non-destructively)."""
    path = path or Path.home() / ".pi" / "agent" / "models.json"
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text() or "{}")
        except json.JSONDecodeError:
            ui.fail(f"{path} is not valid JSON; fix it before launching pi")
    providers = data.setdefault("providers", {})
    prov = providers.setdefault("mlxh", {})
    prov.update({
        "baseUrl": f"http://127.0.0.1:{port}/v1",
        "api": "openai-completions",
        "apiKey": "mlxh",
        # plain OpenAI-compatible server: no developer role / reasoning_effort
        "compat": {"supportsDeveloperRole": False,
                   "supportsReasoningEffort": False},
    })
    models = prov.setdefault("models", [])
    entry = next((item for item in models if item.get("id") == model), None)
    if entry is None:
        entry = {"id": model}
        models.append(entry)
    entry["input"] = ["text", "image"] if supports_images else ["text"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def _stop_owned_server(process):
    """Stop only the isolated server process group this CLI started."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)
    except ProcessLookupError:
        pass


def _ensure_local_server(cfg, name, path, port, *, require_same_model=False,
                         require_chat_protocol=False):
    """Return (info, owned_process), starting an isolated server if absent."""
    try:
        info = _fetch_info(port)
    except Exception:
        info = None
    started = None
    if info is None:
        ui.step(f"starting mlxh serve {name} on port {port}")
        log_path = HOME / "serve.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log:
            started = subprocess.Popen(
                serve_argv(cfg, name, path, {"port": port}),
                stdout=log, stderr=log, start_new_session=True,
            )
        deadline = time.monotonic() + 180
        while True:
            try:
                info = _fetch_info(port)
                runtime = info.get("runtime") or {}
                if runtime.get("ready"):
                    break
            except Exception:
                pass
            if started.poll() is not None or time.monotonic() > deadline:
                _stop_owned_server(started)
                ui.fail("server failed to start", f"see {log_path}")
            time.sleep(0.25)
    serving = info.get("model")
    if serving != name:
        if require_same_model:
            if started:
                _stop_owned_server(started)
            ui.fail(
                f"server on port {port} is serving '{serving}', not '{name}'",
                "stop it or choose the model it already serves",
            )
        ui.note(f"reusing running server on port {port} "
                f"(serving '{serving}', not '{name}')")
    elif started is None:
        ui.note(f"reusing running server on port {port}")
    if require_chat_protocol and (
        info.get("capabilities", {}).get("chat_protocol") != 1
        or info.get("runtime", {}).get("engine_version") != 1
    ):
        if started:
            _stop_owned_server(started)
        ui.fail("the running mlxh server is too old for terminal chat",
                "restart the server and try again")
    return info, started


def cmd_launch(args):
    import shutil as _shutil

    cfg = load_config()
    if args.agent not in AGENTS:
        ui.fail(f"unknown agent '{args.agent}'",
                f"supported: {', '.join(AGENTS)}")

    # argparse REMAINDER swallows our own flags when they follow the agent
    # name; reclaim them. Everything after "--" belongs to the agent.
    rest, cleaned, i = list(args.rest), [], 0
    while i < len(rest):
        a = rest[i]
        if a == "--":
            cleaned.extend(rest[i + 1:])
            break
        if a == "--model" and args.model is None and i + 1 < len(rest):
            args.model = rest[i + 1]
            i += 2
            continue
        if a == "--port" and args.port is None and i + 1 < len(rest):
            args.port = int(rest[i + 1])
            i += 2
            continue
        if a == "--dry-run":
            args.dry_run = True
            i += 1
            continue
        if a == "--no-mcp":
            args.no_mcp = True
            i += 1
            continue
        cleaned.append(a)
        i += 1
    args.rest = cleaned

    name = args.model or pick_model(cfg, f"use with {args.agent}")
    path = resolve(cfg, name)
    _require_language(path)
    port = args.port or cfg["port"]
    env_extra, agent_args = AGENTS[args.agent](port, name)
    if args.no_mcp:
        if args.agent != "claude":
            ui.fail("--no-mcp only applies to claude")
        # strip MCP tool definitions: they add tens of thousands of prompt
        # tokens that local models can't afford
        agent_args += ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']

    if args.dry_run:
        for k, v in env_extra.items():
            print(f"export {k}={v!r}")
        print(" ".join([args.agent, *agent_args, *args.rest]))
        return

    info, started = _ensure_local_server(
        cfg, name, path, port, require_same_model=False,
        require_chat_protocol=False,
    )
    if info.get("model_kind") == "image":
        ui.fail("the running server generates images and cannot serve a coding agent")
    if started is None:
        # A running server keeps the settings it started with; warn when the
        # config has moved on (the classic: raising max_prompt_tokens after
        # the server was already up).
        try:
            live = info["settings"]
            stale = {k: (live[k], cfg[k]) for k in live
                     if k in cfg and live[k] != cfg[k]}
            if stale:
                for k, (have, want) in stale.items():
                    ui.note(f"warning: running server has {k}={have}, config says {want}")
                ui.note("restart the server to apply the new settings")
        except Exception:
            pass

    if args.agent == "pi":
        registry = pi_register_provider(
            port, name, supports_images=model_supports_images(path)
        )
        ui.note(f"registered provider 'mlxh' in {registry}")

    if not _shutil.which(args.agent):
        if started:
            _stop_owned_server(started)
        ui.fail(f"'{args.agent}' is not installed",
                hint="install it first, or use --dry-run to see the wiring")
    cap = cfg["max_prompt_tokens"]
    if cap and cap < 32768 and not args.no_mcp:
        ui.note(f"note: coding agents send ~30k-token prompts (more with MCP servers); "
                f"max_prompt_tokens={cap} will reject them. Raise with "
                f"`mlxh config max_prompt_tokens 40960`, or shrink the prompt with "
                f"`--no-mcp` (claude only).")
    ui.step(f"launching {args.agent} against {name}")
    try:
        proc = subprocess.run([args.agent, *agent_args, *args.rest],
                              env={**os.environ, **env_extra})
    finally:
        if started:
            _stop_owned_server(started)
            ui.note("stopped the mlxh server it started")
    sys.exit(proc.returncode)


def cmd_chat(args):
    cfg = load_config()
    name = args.name or pick_model(cfg, "chat with")
    path = resolve(cfg, name)
    _require_language(path)
    port = cfg["port"]
    _info, started = _ensure_local_server(
        cfg, name, path, port, require_same_model=True,
        require_chat_protocol=True,
    )
    from . import chat_cli

    previous = {}

    def stop_for_signal(signum, _frame):
        _stop_owned_server(started)
        raise SystemExit(128 + signum)

    if started:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, stop_for_signal)
    try:
        chat_cli.run([
            "--base-url", f"http://127.0.0.1:{port}", "--model", name,
            *_chat_args(cfg, args.rest),
        ])
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if started:
            _stop_owned_server(started)
            ui.note("stopped the mlxh server it started")


def _pick_image_model(cfg):
    available = []
    for name, path in discover(cfg).items():
        try:
            if image_metadata(path):
                available.append(name)
        except ValueError as exc:
            ui.note(f"skipping {name}: unsupported image metadata ({exc})")
    if not available:
        ui.fail("no supported image models installed",
                hint="mlxh pull black-forest-labs/FLUX.2-klein-4B --kind image")
    if len(available) == 1:
        ui.note(f"using {available[0]}")
        return available[0]
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        ui.fail("which image model do you want to use?",
                f"available: {', '.join(available)}")
    paths = discover(cfg)
    idx = ui.select("select an image model  (↑/↓, enter)", available,
                    [source_of(paths[name]) for name in available])
    return available[idx]


def _image_api_request(port, model, prompt, *, size, seed, steps, output_format,
                       input_images=None):
    import base64
    import mimetypes
    import secrets
    import urllib.error
    import urllib.request

    body = {"model": model, "prompt": prompt, "size": size,
            "output_format": output_format}
    if seed is not None:
        body["seed"] = seed
    if steps is not None:
        body["steps"] = steps
    if input_images:
        boundary = "mlxh-" + secrets.token_hex(16)
        chunks = []

        def field(name, value):
            chunks.extend((
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(),
                str(value).encode(), b"\r\n",
            ))

        for key, value in body.items():
            field(key, value)
        for path in input_images:
            path = Path(path).expanduser()
            try:
                image_bytes = path.read_bytes()
            except OSError as exc:
                raise RuntimeError(f"could not read reference image {path}: {exc}") from None
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            chunks.extend((
                (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"reference\"\r\n"
                 f"Content-Type: {content_type}\r\n\r\n").encode(),
                image_bytes, b"\r\n",
            ))
        chunks.append(f"--{boundary}--\r\n".encode())
        url = f"http://127.0.0.1:{port}/v1/images/edits"
        payload = b"".join(chunks)
        content_type = f"multipart/form-data; boundary={boundary}"
    else:
        url = f"http://127.0.0.1:{port}/v1/images/generations"
        payload = json.dumps(body).encode()
        content_type = "application/json"
    request = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            error = json.loads(exc.read()).get("error", {}).get("message", str(exc))
        except (ValueError, AttributeError):
            error = str(exc)
        raise ImageAPIError(exc.code, f"image API returned HTTP {exc.code}: {error}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach the image server: {exc.reason}") from None
    return (base64.b64decode(result["data"][0]["b64_json"]),
            result.get("size", size), result.get("mlxh", {}))


class ImageAPIError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _save_cli_image(data, prompt, output_dir, output_format, explicit=None, force=False):
    import re

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if explicit:
        target = Path(explicit).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force:
            ui.fail(f"refusing to overwrite {target}", hint="use --force to replace it")
        try:
            with target.open("xb") as image_file:
                image_file.write(data)
        except FileExistsError:
            if not force:
                ui.fail(f"refusing to overwrite {target}", hint="use --force to replace it")
            target.write_bytes(data)
        return target

    stem = re.sub(r"[^\w-]+", "-", prompt.casefold(), flags=re.UNICODE).strip("-_")[:64].rstrip("-_")
    stem = stem or "image"
    extension = "jpg" if output_format == "jpeg" else output_format
    for number in range(1, 1_000_000):
        suffix = "" if number == 1 else f"-{number}"
        target = output_dir / f"{stem}{suffix}.{extension}"
        try:
            with target.open("xb") as image_file:
                image_file.write(data)
            return target
        except FileExistsError:
            continue
    ui.fail(f"could not find an unused filename for {stem}.{extension}")


def _image_help():
    print("Enter a prompt to generate; /ref PATH attaches an image for the next prompt. "
          "Commands: /clear-refs, /size auto|WIDTHxHEIGHT, /seed random|N, "
          "/steps default|1..100, /format png|jpeg|webp, /output DIR, /help, /exit")


def _image_session(input=None, output=None):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion

    commands = ("/ref", "/clear-refs", "/size", "/seed", "/steps",
                "/format", "/output", "/help", "/exit")

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/") or " " in text:
                return
            for command in commands:
                if command.startswith(text):
                    yield Completion(command, start_position=-len(text))

    return PromptSession(completer=SlashCompleter(), complete_while_typing=False,
                         input=input, output=output)


def cmd_image(args):
    import re

    cfg = load_config()
    name = args.model or _pick_image_model(cfg)
    path = resolve(cfg, name)
    if checked_model_kind(path) != "image":
        ui.fail(f"'{name}' is not an image generation model")
    prompts = " ".join(args.prompt).strip()
    from .images import MAX_REFERENCE_IMAGE_BYTES, MAX_REFERENCE_IMAGES
    input_images = [Path(image).expanduser().resolve()
                    for image in (getattr(args, "input_image", None) or [])]
    if len(input_images) > MAX_REFERENCE_IMAGES:
        ui.fail(f"at most {MAX_REFERENCE_IMAGES} reference images may be attached")
    for image in input_images:
        if not image.is_file() or not os.access(image, os.R_OK):
            ui.fail(f"reference image does not exist or is unreadable: {image}")
        if image.stat().st_size > MAX_REFERENCE_IMAGE_BYTES:
            ui.fail(f"reference image exceeds the 25-MiB limit: {image}")
    if args.output and not prompts:
        ui.fail("--output requires a one-shot prompt")
    if args.force and not args.output:
        ui.fail("--force requires --output")
    if args.output and Path(args.output).expanduser().exists() and not args.force:
        ui.fail(f"refusing to overwrite {Path(args.output).expanduser()}",
                hint="use --force to replace it")

    port = cfg["port"]
    info, started = _ensure_local_server(
        cfg, name, path, port, require_same_model=True,
    )
    if info.get("model_kind") != "image":
        if started:
            _stop_owned_server(started)
        ui.fail("the running server is not an image generation server")
    if input_images and not info.get("capabilities", {}).get("image_edits", False):
        if started:
            _stop_owned_server(started)
        ui.fail(f"'{name}' does not support image editing")

    previous = {}

    def stop_for_signal(signum, _frame):
        _stop_owned_server(started)
        raise SystemExit(128 + signum)

    if started:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, stop_for_signal)

    state = {
        "size": args.size, "seed": args.seed, "steps": args.steps,
        "format": args.output_format, "output_dir": Path(args.output_dir or Path.cwd()),
        "references": [str(image) for image in input_images],
    }

    def generate(prompt, explicit=None, force=False):
        ui.step("generating image...")
        reference_count = len(state["references"])
        try:
            request_options = {}
            if state["references"]:
                request_options["input_images"] = state["references"]
            data, resolved_size, details = _image_api_request(
                port, name, prompt, size=state["size"], seed=state["seed"],
                steps=state["steps"], output_format=state["format"],
                **request_options,
            )
        except ImageAPIError as exc:
            if exc.status in (499, 500, 504):
                state["references"] = []
                if reference_count:
                    exc.args = (f"{exc} (reference images were consumed; attach them again to retry)",)
            raise
        state["references"] = []
        target = _save_cli_image(
            data, prompt, state["output_dir"], state["format"], explicit, force,
        )
        seed_text = f", seed {details['seed']}" if details.get("seed") is not None else ""
        steps_text = f", {details['steps']} steps" if details.get("steps") is not None else ""
        edit_text = f", edited {reference_count} reference(s)" if reference_count else ""
        ui.ok(f"saved {target} ({resolved_size}{seed_text}{steps_text}{edit_text})")

    try:
        if prompts:
            generate(prompts, args.output, args.force)
            return
        if not sys.stdin.isatty():
            ui.fail("provide a prompt or run `mlxh image` in a terminal")
        _image_help()
        session = _image_session()
        while True:
            try:
                line = session.prompt("image> ").strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            if not line.startswith("/"):
                try:
                    generate(line)
                except RuntimeError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                continue
            command, _, value = line.partition(" ")
            value = value.strip()
            try:
                if command in ("/exit", "/quit"):
                    break
                if command == "/help":
                    _image_help()
                elif command == "/ref":
                    image = Path(value).expanduser().resolve()
                    if not value or not image.is_file() or not os.access(image, os.R_OK):
                        print("usage: /ref PATH (path must name a readable local image)")
                    elif len(state["references"]) >= MAX_REFERENCE_IMAGES:
                        print(f"at most {MAX_REFERENCE_IMAGES} reference images may be attached")
                    elif image.stat().st_size > MAX_REFERENCE_IMAGE_BYTES:
                        print(f"reference image exceeds the 25-MiB limit: {image}")
                    elif not info.get("capabilities", {}).get("image_edits", False):
                        print(f"{name} does not support image editing")
                    else:
                        state["references"].append(str(image))
                        ui.ok(f"[Image #{len(state['references'])}] attached: {image}")
                elif command == "/clear-refs":
                    count = len(state["references"])
                    state["references"] = []
                    ui.note(f"cleared {count} reference image(s)")
                elif command == "/size":
                    if value not in ("auto",) and not re.fullmatch(r"\d{1,4}x\d{1,4}", value):
                        print("usage: /size auto|WIDTHxHEIGHT")
                    else:
                        state["size"] = value
                        ui.note(f"size set to {value}")
                elif command == "/seed":
                    state["seed"] = None if not value or value == "random" else int(value)
                    if state["seed"] is not None and not 0 <= state["seed"] < 2**32:
                        raise ValueError
                    ui.note(f"seed set to {state['seed'] if state['seed'] is not None else 'random'}")
                elif command == "/steps":
                    state["steps"] = None if not value or value == "default" else int(value)
                    if state["steps"] is not None and not 1 <= state["steps"] <= 100:
                        raise ValueError
                    ui.note(f"steps set to {state['steps'] if state['steps'] is not None else 'model default'}")
                elif command == "/format":
                    if value not in ("png", "jpeg", "webp"):
                        print("usage: /format png|jpeg|webp")
                    else:
                        state["format"] = value
                        ui.note(f"format set to {value}")
                elif command == "/output":
                    if not value:
                        print("usage: /output DIR")
                    else:
                        state["output_dir"] = Path(value).expanduser()
                        ui.note(f"output directory set to {state['output_dir']}")
                else:
                    print("unknown command; use /help")
            except ValueError:
                print("invalid value; use /help for command syntax")
    except RuntimeError as exc:
        ui.fail(str(exc))
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if started:
            _stop_owned_server(started)
            ui.note("stopped the mlxh server it started")


def cmd_config(args):
    cfg = load_config()
    if not args.key:
        print(json.dumps(cfg, indent=2))
        return
    if args.value is None:
        print(cfg.get(args.key))
        return
    if args.key not in KEY_TYPES:
        ui.fail(f"unknown config key '{args.key}'",
                f"settable: {', '.join(KEY_TYPES)}")
    try:
        cfg[args.key] = KEY_TYPES[args.key](args.value)
    except ValueError:
        kind = "bool" if KEY_TYPES[args.key] is _bool else KEY_TYPES[args.key].__name__
        ui.fail(f"'{args.key}' expects a {kind}")
    save_config(cfg)
    ui.ok(f"{args.key} = {cfg[args.key]}")


AGENT_MENU = [
    ("Launch Claude Code", "claude", "Anthropic's coding agent (core tools, no MCP)"),
    ("Launch Codex", "codex", "OpenAI's coding agent"),
    ("Launch pi", "pi", "pi coding agent"),
]


def cmd_home(parser):
    """Bare `mlxh`: an interactive home menu instead of an argparse error."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        parser.print_help()
        return
    import shutil as _shutil
    from types import SimpleNamespace as NS

    from . import __version__
    print(ui.bold(f"mlxh {__version__}"))
    labels, notes, actions = [], [], []
    for label, agent, note in AGENT_MENU:
        installed = _shutil.which(agent)
        labels.append(label)
        notes.append(note if installed else f"{note} — not installed")
        actions.append(("launch", agent))
    labels += ["Chat", "Generate image", "Serve", "List models"]
    notes += ["talk to a model in this terminal",
              "generate with an installed image model",
              "OpenAI + Anthropic API server",
              "what's installed, sizes, sources"]
    actions += [("chat", None), ("image", None), ("serve", None), ("list", None)]

    idx = ui.select("what do you want to do?  (↑/↓ + enter, q quits)", labels, notes)
    kind, agent = actions[idx]
    if kind == "launch":
        cmd_launch(NS(agent=agent, model=None, port=None, dry_run=False,
                      no_mcp=(agent == "claude"), rest=[]))
    elif kind == "chat":
        cmd_chat(NS(name=None, rest=[]))
    elif kind == "image":
        cmd_image(NS(model=None, prompt=[], output=None, output_dir=None,
                     force=False, size="auto", seed=None, steps=None,
                     output_format="png"))
    elif kind == "list":
        cmd_list(None)
    else:
        cmd_serve(NS(name=None, port=None, host=None, max_queued=None,
                     max_tokens_cap=None, memory_limit_gb=None,
                     cache_limit_gb=None, gen_timeout_s=None,
                     max_prompt_tokens=None, prompt_cache=None, thinking=None,
                     max_image_pixels=None, image_steps=None))


def cmd_uninstall(args):
    cfg = load_config()
    root = models_dir(cfg)
    outside = not str(root.resolve()).startswith(str(HOME.resolve()))
    if not args.yes:
        extra = f" Models at {root} are OUTSIDE {HOME} and will be kept." if outside else ""
        reply = input(
            f"This deletes {HOME} (venv, config, models, caches) and the mlxh "
            f"launcher.{extra} Type 'yes' to continue: "
        ).strip().lower()
        if reply != "yes":
            sys.exit("aborted")
    launcher = os.environ.get("MLXH_LAUNCHER")
    if _service_plist_path().exists():
        _service_uninstall()
        ui.note("stopped and removed the login service")
    shutil.rmtree(HOME, ignore_errors=True)
    if launcher and Path(launcher).is_file():
        Path(launcher).unlink()
        print(f"removed {launcher}")
    print(f"removed {HOME}." + (f" (kept {root})" if outside else "") + " Goodbye.")


def main():
    ap = argparse.ArgumentParser(prog="mlxh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=False)

    p = sub.add_parser("search", help="search Hugging Face for MLX models")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("run", help="chat with a model, pulling it first if needed")
    p.add_argument("target", nargs="?",
                   help="installed model name or Hugging Face repo id")
    p.add_argument("--force", action="store_true",
                   help="download even if it exceeds this machine's memory")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("pull", help="download a model from Hugging Face")
    p.add_argument("repo")
    p.add_argument("--name")
    p.add_argument("--kind", choices=["auto", "language", "image"], default="auto")
    p.add_argument("--backend", choices=["mflux"])
    p.add_argument("--force", action="store_true",
                   help="download even if it exceeds this machine's memory")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("images", help="manage optional image generation support")
    p.add_argument("action", choices=["install"])
    p.set_defaults(fn=cmd_images)

    p = sub.add_parser("link", help="symlink an existing local model directory in")
    p.add_argument("path")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_link)

    p = sub.add_parser("list", help="show available models")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("mv", help="rename a model")
    p.add_argument("name")
    p.add_argument("new_name")
    p.set_defaults(fn=cmd_mv)

    p = sub.add_parser("rm", help="remove a model (links: symlink only)")
    p.add_argument("name")
    p.set_defaults(fn=cmd_rm)

    p = sub.add_parser("serve", help="run the OpenAI-compatible API server")
    p.add_argument("name", nargs="?")
    p.add_argument("--port", type=int)
    p.add_argument("--host")
    p.add_argument("--max-queued", type=int, dest="max_queued")
    p.add_argument("--max-tokens-cap", type=int, dest="max_tokens_cap")
    p.add_argument("--memory-limit-gb", type=float, dest="memory_limit_gb")
    p.add_argument("--cache-limit-gb", type=float, dest="cache_limit_gb")
    p.add_argument("--gen-timeout-s", type=int, dest="gen_timeout_s")
    p.add_argument("--max-prompt-tokens", type=int, dest="max_prompt_tokens")
    p.add_argument("--prompt-cache", dest="prompt_cache")
    p.add_argument("--thinking", choices=["auto", "on", "off"], dest="thinking")
    p.add_argument("--max-image-pixels", type=int)
    p.add_argument("--image-steps", type=int)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("status", help="show live stats of the running server")
    p.add_argument(
        "--json", action="store_true",
        help="emit JSON (use set -o pipefail when piping to another command)",
    )
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("service", help="manage the persistent login server")
    p.add_argument("action", choices=["install", "uninstall", "restart"])
    p.add_argument("--dry-run", action="store_true",
                   help="print the plist and launchctl commands without changes")
    p.set_defaults(fn=cmd_service)

    p = sub.add_parser("launch", help="launch a coding agent (claude, codex, pi) on a local model")
    p.add_argument("agent", help="claude, codex, or pi")
    p.add_argument("--model")
    p.add_argument("--port", type=int)
    p.add_argument("--dry-run", action="store_true", help="print env + command instead of running")
    p.add_argument("--no-mcp", action="store_true",
                   help="claude: launch without MCP servers (much smaller prompts)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_launch)

    p = sub.add_parser("chat", help="terminal chat (extra args go to the chat CLI)")
    p.add_argument("name", nargs="?")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("image", help="generate images interactively or from one prompt")
    p.add_argument("model", nargs="?", help="installed image model (prompts to choose if omitted)")
    p.add_argument("prompt", nargs="*", help="prompt text; omit for interactive mode")
    p.add_argument("--output", help="exact output path for one-shot generation (never overwrites by default)")
    p.add_argument("--input-image", action="append", default=[], metavar="PATH",
                   help="reference image for editing; may be repeated (edit-capable models only)")
    p.add_argument("--output-dir", help="directory for generated images (default: current directory)")
    p.add_argument("--force", action="store_true", help="allow overwriting an explicit --output path")
    p.add_argument("--size", default="auto", help="auto or WIDTHxHEIGHT (default: auto)")
    p.add_argument("--seed", type=int, help="fixed seed (default: random)")
    p.add_argument("--steps", type=int, help="denoising steps (default: model default)")
    p.add_argument("--output-format", choices=["png", "jpeg", "webp"], default="png")
    p.set_defaults(fn=cmd_image)

    p = sub.add_parser("config", help="show or set config")
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("uninstall", help="remove mlxh entirely")
    p.add_argument("--yes", action="store_true", help="skip confirmation")
    p.set_defaults(fn=cmd_uninstall)

    args = ap.parse_args()
    try:
        if args.cmd is None:
            cmd_home(ap)
        else:
            args.fn(args)
    except KeyboardInterrupt:
        print()
        sys.exit(130)


if __name__ == "__main__":
    main()
