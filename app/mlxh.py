"""mlxh — a small harness for running local MLX models with an OpenAI API.

Commands:
  mlxh pull <hf-repo> [--name NAME]   download a model from Hugging Face
  mlxh link <path> [--name NAME]      register an existing local model dir
  mlxh list                           show registered models
  mlxh rm <name>                      remove a model (deletes only pulled files)
  mlxh serve <name> [--port N]        OpenAI-compatible API server
  mlxh chat <name> [chat args...]     terminal chat (tools, images, streaming)
  mlxh config [key [value]]           show or set config (port, host)
  mlxh uninstall                      remove mlxh and everything it manages

State lives under $MLXH_HOME (default ~/.mlxh): venv, app code, config,
downloaded models, HF cache. Uninstall removes exactly that plus the launcher.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(os.environ.get("MLXH_HOME", Path.home() / ".mlxh"))
CONFIG = HOME / "config.json"
MODELS = HOME / "models"
APP = Path(__file__).parent
DEFAULTS = {"port": 8081, "host": "127.0.0.1", "models": {}}


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG.exists():
        cfg.update(json.loads(CONFIG.read_text()))
    return cfg


def save_config(cfg):
    HOME.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2))


def resolve(cfg, name):
    path = cfg["models"].get(name)
    if not path or not Path(path).is_dir():
        registered = ", ".join(sorted(cfg["models"])) or "(none)"
        sys.exit(f"unknown or missing model '{name}'. Registered: {registered}")
    return path


def cmd_pull(args):
    cfg = load_config()
    name = args.name or args.repo.split("/")[-1]
    dest = MODELS / name
    if dest.exists():
        sys.exit(f"model '{name}' already exists ({dest}); pick --name or `mlxh rm {name}` first")
    MODELS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HOME / "hf-cache"))
    from huggingface_hub import snapshot_download
    print(f"Downloading {args.repo} -> {dest}")
    snapshot_download(args.repo, local_dir=str(dest))
    cfg["models"][name] = str(dest)
    save_config(cfg)
    print(f"Registered '{name}'. Try: mlxh chat {name}")


def cmd_link(args):
    cfg = load_config()
    path = Path(args.path).expanduser().resolve()
    if not (path / "config.json").is_file():
        sys.exit(f"{path} does not look like a model directory (no config.json)")
    name = args.name or path.name
    cfg["models"][name] = str(path)
    save_config(cfg)
    print(f"Registered '{name}' -> {path} (linked; `mlxh rm` will only unregister it)")


def cmd_list(_args):
    cfg = load_config()
    if not cfg["models"]:
        print("no models registered. Try: mlxh pull <hf-repo>")
        return
    for name, path in sorted(cfg["models"].items()):
        kind = "pulled" if Path(path).is_relative_to(MODELS) else "linked"
        size = ""
        if Path(path).is_dir():
            n = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
            size = f"{n / 1e9:.1f} GB"
        print(f"{name:24} {kind:7} {size:>9}  {path}")


def cmd_rm(args):
    cfg = load_config()
    path = Path(resolve(cfg, args.name))
    if path.is_relative_to(MODELS):
        shutil.rmtree(path)
        print(f"deleted {path}")
    else:
        print(f"unregistered '{args.name}' (linked files at {path} untouched)")
    del cfg["models"][args.name]
    save_config(cfg)


def cmd_serve(args):
    cfg = load_config()
    path = resolve(cfg, args.name)
    port = args.port or cfg["port"]
    os.execv(sys.executable, [
        sys.executable, str(APP / "serve_app.py"),
        "--model-path", path, "--name", args.name,
        "--port", str(port), "--host", args.host or cfg["host"],
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
        print(json.dumps({k: v for k, v in cfg.items() if k != "models"}, indent=2))
        return
    if args.value is None:
        print(cfg.get(args.key))
        return
    value = args.value
    if args.key not in DEFAULTS or args.key == "models":
        sys.exit(f"unknown config key '{args.key}' (settable: port, host)")
    cfg[args.key] = int(value) if args.key == "port" else value
    save_config(cfg)
    print(f"{args.key} = {cfg[args.key]}")


def cmd_uninstall(args):
    n_models = len(load_config()["models"])
    if not args.yes:
        reply = input(
            f"This deletes {HOME} (venv, config, {n_models} registered model(s) — "
            "pulled weights included) and the mlxh launcher. Type 'yes' to continue: "
        ).strip().lower()
        if reply != "yes":
            sys.exit("aborted")
    launcher = os.environ.get("MLXH_LAUNCHER")
    shutil.rmtree(HOME, ignore_errors=True)
    if launcher and Path(launcher).is_file():
        Path(launcher).unlink()
        print(f"removed {launcher}")
    print(f"removed {HOME}. Goodbye.")


def main():
    ap = argparse.ArgumentParser(prog="mlxh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="download a model from Hugging Face")
    p.add_argument("repo")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("link", help="register an existing local model directory")
    p.add_argument("path")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_link)

    p = sub.add_parser("list", help="show registered models")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("rm", help="remove a model")
    p.add_argument("name")
    p.set_defaults(fn=cmd_rm)

    p = sub.add_parser("serve", help="run the OpenAI-compatible API server")
    p.add_argument("name")
    p.add_argument("--port", type=int)
    p.add_argument("--host")
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
