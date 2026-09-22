"""mlxh — a small harness for running local MLX models with an OpenAI API.

Commands:
  mlxh pull <hf-repo> [--name NAME]   download a model from Hugging Face
  mlxh link <path> [--name NAME]      symlink an existing local model dir in
  mlxh list                           show available models
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

HOME = Path(os.environ.get("MLXH_HOME", Path.home() / ".mlxh"))
CONFIG = HOME / "config.json"
APP = Path(__file__).parent
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


def resolve(cfg, name):
    path = models_dir(cfg) / name
    if not is_model(path):
        names = ", ".join(discover(cfg)) or "(none)"
        sys.exit(f"no model '{name}' in {models_dir(cfg)}. Available: {names}")
    return str(path)


def cmd_pull(args):
    cfg = load_config()
    name = args.name or args.repo.split("/")[-1]
    dest = models_dir(cfg) / name
    if dest.exists():
        sys.exit(f"'{name}' already exists ({dest}); pick --name or `mlxh rm {name}` first")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HOME / "hf-cache"))
    from huggingface_hub import snapshot_download
    print(f"Downloading {args.repo} -> {dest}")
    snapshot_download(args.repo, local_dir=str(dest))
    print(f"Done. Try: mlxh chat {name}")


def cmd_link(args):
    cfg = load_config()
    target = Path(args.path).expanduser().resolve()
    if not is_model(target):
        sys.exit(f"{target} does not look like a model directory (no config.json)")
    name = args.name or target.name
    dest = models_dir(cfg) / name
    if dest.exists() or dest.is_symlink():
        sys.exit(f"'{name}' already exists ({dest})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(target)
    print(f"Linked '{name}' -> {target} (`mlxh rm {name}` removes only the link)")


def cmd_list(_args):
    cfg = load_config()
    models = discover(cfg)
    if not models:
        print(f"no models in {models_dir(cfg)}. Try: mlxh pull <hf-repo>")
        return
    for name, path in models.items():
        kind = "linked" if path.is_symlink() else "pulled"
        n = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"{name:24} {kind:7} {n / 1e9:7.1f} GB  {path.resolve() if path.is_symlink() else path}")


def cmd_rm(args):
    cfg = load_config()
    path = models_dir(cfg) / args.name
    if path.is_symlink():
        target = path.resolve()
        path.unlink()
        print(f"removed link '{args.name}' (files at {target} untouched)")
    elif is_model(path):
        shutil.rmtree(path)
        print(f"deleted {path}")
    else:
        sys.exit(f"no model '{args.name}' in {models_dir(cfg)}")


def cmd_serve(args):
    cfg = load_config()
    path = resolve(cfg, args.name)

    def pick(cli_value, key):
        return cli_value if cli_value is not None else cfg[key]

    os.execv(sys.executable, [
        sys.executable, str(APP / "serve_app.py"),
        "--model-path", path, "--name", args.name,
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
    path = resolve(cfg, args.name)
    os.execv(sys.executable, [
        sys.executable, str(APP / "chat_cli.py"), "--model-path", path, *args.rest,
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
        sys.exit(f"unknown config key '{args.key}' (settable: {', '.join(KEY_TYPES)})")
    try:
        cfg[args.key] = KEY_TYPES[args.key](args.value)
    except ValueError:
        sys.exit(f"'{args.key}' expects a {KEY_TYPES[args.key].__name__}")
    save_config(cfg)
    print(f"{args.key} = {cfg[args.key]}")


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

    p = sub.add_parser("pull", help="download a model from Hugging Face")
    p.add_argument("repo")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("link", help="symlink an existing local model directory in")
    p.add_argument("path")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_link)

    p = sub.add_parser("list", help="show available models")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("rm", help="remove a model (links: symlink only)")
    p.add_argument("name")
    p.set_defaults(fn=cmd_rm)

    p = sub.add_parser("serve", help="run the OpenAI-compatible API server")
    p.add_argument("name")
    p.add_argument("--port", type=int)
    p.add_argument("--host")
    p.add_argument("--max-queued", type=int, dest="max_queued")
    p.add_argument("--max-tokens-cap", type=int, dest="max_tokens_cap")
    p.add_argument("--memory-limit-gb", type=float, dest="memory_limit_gb")
    p.add_argument("--cache-limit-gb", type=float, dest="cache_limit_gb")
    p.add_argument("--gen-timeout-s", type=int, dest="gen_timeout_s")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("chat", help="terminal chat (extra args go to the chat CLI)")
    p.add_argument("name")
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
