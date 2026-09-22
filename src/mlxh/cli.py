"""mlxh — a small harness for running local MLX models with an OpenAI API.

Commands:
  mlxh run <repo-or-name>             chat now, pulling first if needed
  mlxh pull <hf-repo> [--name NAME]   download a model from Hugging Face
  mlxh link <path> [--name NAME]      symlink an existing local model dir in
  mlxh list                           show available models
  mlxh mv <name> <new-name>           rename a model
  mlxh rm <name>                      remove a model (links: symlink only)
  mlxh serve <name> [--port N ...]    OpenAI-compatible API server
  mlxh chat <name> [chat args...]     terminal chat (tools, images, streaming)
  mlxh config [key [value]]           show or set config
  mlxh uninstall                      remove mlxh and everything it manages

Models live in ONE directory and the filesystem is the registry: every
subdirectory of the models dir that holds a config.json is a usable model.
`pull` downloads there; `link` drops a symlink there. The location is the
`models_dir` config key, overridable with $MLXH_MODELS_DIR.

Config keys (mlxh config <key> <value>):
  port, host          server defaults
  models_dir          where models live (default $MLXH_HOME/models)
  max_queued          pending generations beyond the active one before 503 (4)
  max_tokens_cap      server-side ceiling on max_tokens, 0 = unlimited (16384)
  memory_limit_gb     MLX GPU/unified-memory limit, 0 = off
  cache_limit_gb      MLX buffer-cache limit, 0 = off
  gen_timeout_s       hard stop for one generation, 0 = off (600)

State lives under $MLXH_HOME (default ~/.mlxh): venv, app code, config,
models, HF cache. Uninstall removes exactly that plus the launcher.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from . import ui

HOME = Path(os.environ.get("MLXH_HOME", Path.home() / ".mlxh"))
CONFIG = HOME / "config.json"
DEFAULTS = {
    "port": 8081,
    "host": "127.0.0.1",
    "models_dir": str(HOME / "models"),
    "max_queued": 4,
    "max_tokens_cap": 16384,
    "memory_limit_gb": 0.0,
    "cache_limit_gb": 0.0,
    "gen_timeout_s": 600,
}
KEY_TYPES = {
    "port": int, "host": str, "models_dir": str, "max_queued": int,
    "max_tokens_cap": int, "memory_limit_gb": float, "cache_limit_gb": float,
    "gen_timeout_s": int,
}


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
    return (path / "config.json").is_file()


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
    print(ui.dim(f"select a model to {purpose}:"))
    for i, n in enumerate(models, 1):
        print(f"  {i}) {n}")
    while True:
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(130)
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        if choice in models:
            return choice
        print(ui.dim(f"enter 1-{len(models)} or a model name"))


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


def do_pull(cfg, repo, name, force=False):
    dest = models_dir(cfg) / name
    if dest.exists():
        ui.fail(f"'{name}' already exists",
                f"at {dest}",
                hint=f"pick another with --name, or `mlxh rm {name}` first")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HOME / "hf-cache"))
    from datetime import datetime, timezone
    from huggingface_hub import HfApi, snapshot_download

    try:
        size = sum(s.size or 0 for s in
                   HfApi().model_info(repo, files_metadata=True).siblings)
    except Exception:
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
    try:
        snapshot_download(repo, local_dir=str(dest))
    except Exception as e:
        shutil.rmtree(dest, ignore_errors=True)
        ui.fail("download failed",
                f"{type(e).__name__}: {e}",
                hint="check the repo id with `mlxh search`; gated repos need `hf auth login`")
    try:
        revision = HfApi().model_info(repo).sha or ""
    except Exception:
        revision = ""
    (dest / ".mlxh.json").write_text(json.dumps({
        "repo": repo,
        "revision": revision,
        "pulled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2))
    ui.ok(f"pulled {repo}@{revision[:7]} as '{name}'")


def cmd_pull(args):
    cfg = load_config()
    name = args.name or args.repo.split("/")[-1]
    do_pull(cfg, args.repo, name, force=args.force)
    ui.note(f"chat: mlxh chat {name}    serve: mlxh serve {name}")


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
                do_pull(cfg, args.target, name, force=args.force)
    path = resolve(cfg, name)
    os.execv(sys.executable, [
        sys.executable, "-m", "mlxh.chat_cli", "--model-path", path, *args.rest,
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
    print(ui.dim(f"{'NAME':{nw}}  {'KIND':6} {'SIZE':>8}  SOURCE"))
    home = str(Path.home())
    for name, kind, size, source, target in rows:
        line = f"{name:{nw}}  {kind:6} {size:>8}  {source:{sw}}"
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


def cmd_serve(args):
    cfg = load_config()
    name = args.name or pick_model(cfg, "serve")
    path = resolve(cfg, name)

    def pick(cli_value, key):
        return cli_value if cli_value is not None else cfg[key]

    os.execv(sys.executable, [
        sys.executable, "-m", "mlxh.serve_app",
        "--model-path", path, "--name", name,
        "--port", str(pick(args.port, "port")),
        "--host", str(pick(args.host, "host")),
        "--max-queued", str(pick(args.max_queued, "max_queued")),
        "--max-tokens-cap", str(pick(args.max_tokens_cap, "max_tokens_cap")),
        "--memory-limit-gb", str(pick(args.memory_limit_gb, "memory_limit_gb")),
        "--cache-limit-gb", str(pick(args.cache_limit_gb, "cache_limit_gb")),
        "--gen-timeout-s", str(pick(args.gen_timeout_s, "gen_timeout_s")),
    ])


def cmd_chat(args):
    cfg = load_config()
    path = resolve(cfg, args.name or pick_model(cfg, "chat with"))
    os.execv(sys.executable, [
        sys.executable, "-m", "mlxh.chat_cli", "--model-path", path, *args.rest,
    ])


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
        ui.fail(f"'{args.key}' expects a {KEY_TYPES[args.key].__name__}")
    save_config(cfg)
    ui.ok(f"{args.key} = {cfg[args.key]}")


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
    shutil.rmtree(HOME, ignore_errors=True)
    if launcher and Path(launcher).is_file():
        Path(launcher).unlink()
        print(f"removed {launcher}")
    print(f"removed {HOME}." + (f" (kept {root})" if outside else "") + " Goodbye.")


def main():
    ap = argparse.ArgumentParser(prog="mlxh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

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
    p.add_argument("--force", action="store_true",
                   help="download even if it exceeds this machine's memory")
    p.set_defaults(fn=cmd_pull)

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
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("chat", help="terminal chat (extra args go to the chat CLI)")
    p.add_argument("name", nargs="?")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("config", help="show or set config")
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("uninstall", help="remove mlxh entirely")
    p.add_argument("--yes", action="store_true", help="skip confirmation")
    p.set_defaults(fn=cmd_uninstall)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
