"""Terminal chat for any mlxh-managed model, with built-in tools.

Invoked by `mlxh chat <model>`; also standalone:
    python chat_cli.py --model-path /path/to/model [-p "question"] [-i img.png]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from . import ui
from .loader import load_runner
from .toolcalls import load_user_tools, parse_tool_calls, run_tool

MAX_TOOL_ROUNDS = 5


def generate_once(runner, messages, images, max_tokens, specs, thinking=None):
    """One generation pass; streams visible text, hides tool-call XML."""
    prompt = runner.template(messages, num_images=len(images), tools=specs,
                             thinking=thinking)
    think_open = isinstance(prompt, str) and prompt.rstrip().endswith("<think>")
    parts, printed = [], 0
    marker = "<tool_call>"
    last = None
    rend = ui.StreamRenderer(think_open=think_open)
    for resp in runner.stream(prompt, images=images, max_tokens=max_tokens):
        parts.append(resp.text)
        full = "".join(parts)
        cut = full.find(marker)
        # hold back a tag-length tail so a half-arrived "<tool_ca" never prints
        visible = full[:cut] if cut != -1 else full[: max(0, len(full) - len(marker))]
        if len(visible) > printed:
            rend.feed(visible[printed:])
            printed = len(visible)
        last = resp
    full = "".join(parts)
    cut = full.find(marker)
    visible = full[:cut] if cut != -1 else full
    if len(visible) > printed:
        rend.feed(visible[printed:])
    rend.finish()
    return full, last


def ask(runner, messages, images, max_tokens, tools=None, thinking=None):
    """tools: (registry, specs) to enable the agent loop, or None."""
    registry, specs = tools if tools else ({}, None)
    total_tokens, tps = 0, 0.0
    for _ in range(MAX_TOOL_ROUNDS):
        text, last = generate_once(runner, messages, images, max_tokens, specs,
                                   thinking=thinking)
        total_tokens += last.generation_tokens
        tps = last.generation_tps
        content, tool_calls = parse_tool_calls(text)
        if not tool_calls:
            print("\n\n" + ui.dim(f"[{total_tokens} tokens @ {tps:.1f} tok/s]"))
            messages.append({"role": "assistant", "content": content})
            return content
        messages.append({
            "role": "assistant", "content": content or "",
            "tool_calls": [
                {"function": {"name": tc["function"]["name"],
                              "arguments": json.loads(tc["function"]["arguments"])}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            result = run_tool(registry, name, args)
            print("\n" + ui.dim(f"[tool] {name}({json.dumps(args, ensure_ascii=False)}) -> "
                  f"{json.dumps(result, ensure_ascii=False)}", sys.stderr), file=sys.stderr)
            messages.append({"role": "tool", "content": json.dumps(result)})
        images = []  # images only accompany the first pass
    print("\n(stopped: too many tool rounds)", file=sys.stderr)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("-p", "--prompt", help="one-shot prompt (omit for interactive chat)")
    ap.add_argument("-i", "--image", action="append", default=[], help="image file to include")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tools", action="store_true",
                    help="enable built-in tools (weather, time, calculator)")
    ap.add_argument("--no-tools", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--thinking", action="store_true",
                    help="ask the model to reason before answering (shown dimmed)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="ask the model to skip reasoning")
    args = ap.parse_args()
    thinking = True if args.thinking else (False if args.no_thinking else None)

    print(f"Loading {Path(args.model_path).name}...", file=sys.stderr)
    try:
        runner = load_runner(args.model_path)
    except RuntimeError as e:
        sys.exit(str(e))
    if args.image and not runner.supports_images:
        sys.exit("this model does not support images")
    tools = None
    if args.tools and not args.no_tools:
        registry, specs, tools_path = load_user_tools()
        if registry is None:
            sys.exit(f"--tools: no tools file at {tools_path}\n"
                     "create one from the example: cp examples/tools.py "
                     f"{tools_path}\n"
                     "https://github.com/amenophis1er/mlxh/blob/main/examples/tools.py")
        if not registry:
            sys.exit(f"{tools_path} defines no TOOL_REGISTRY")
        print(f"Tools ({tools_path.name}): {', '.join(registry)}", file=sys.stderr)
        tools = (registry, specs)

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        ask(runner, messages, args.image, args.max_tokens, tools, thinking=thinking)
        return

    prompt_ansi = False
    try:
        import atexit
        import readline
        hist = Path(os.environ.get("MLXH_HOME", Path.home() / ".mlxh")) / "chat_history"
        hist.parent.mkdir(parents=True, exist_ok=True)
        try:
            readline.read_history_file(hist)
        except OSError:
            pass
        readline.set_history_length(500)
        atexit.register(lambda: readline.write_history_file(hist))

        # Tab-complete the slash commands. Without a completer, a Tab on a
        # partial word (e.g. "/exi<Tab>") runs the default filename completion,
        # which inserts stray whitespace and dirties the line.
        cmds = ["/exit", "/reset", "/help", "/bye", "/quit"]
        if runner.supports_images:
            cmds.insert(0, "/image ")

        def completer(text, state):
            hits = [c for c in cmds if c.startswith(text)] if text.startswith("/") else []
            return hits[state] if state < len(hits) else None

        readline.set_completer(completer)
        readline.set_completer_delims(" \t\n")  # treat "/exit" as one word
        is_gnu = "libedit" not in (readline.__doc__ or "")
        readline.parse_and_bind("tab: complete" if is_gnu else "bind ^I rl_complete")
        # The \001/\002 non-printing markers are GNU-readline-only; under
        # macOS libedit they corrupt input accounting, so colorize the prompt
        # only on real GNU readline.
        prompt_ansi = (is_gnu and sys.stdin.isatty() and sys.stdout.isatty()
                       and not os.environ.get("NO_COLOR"))
    except ImportError:
        pass

    hint = "/image <path> attaches an image, " if runner.supports_images else ""
    print(f"Interactive chat. {hint}/reset clears history, /exit quits, /help lists commands.",
          file=sys.stderr)
    history, staged = [], []
    while True:
        tag = f" [{len(staged)} img]" if staged else ""
        if prompt_ansi:
            # \001/\002 tell readline the ANSI codes take no screen width
            prompt = f"\n\001\x1b[1;36m\002you{tag}>\001\x1b[0m\002 "
        else:
            prompt = f"\nyou{tag}> "
        try:
            user = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/bye", "/quit"):
            break
        if user == "/help":
            print("/image <path>  attach an image to your next message\n"
                  "/reset         clear conversation history\n"
                  "/exit          quit (also /bye, /quit, Ctrl-D)", file=sys.stderr)
            continue
        if user == "/reset":
            history, staged = [], []
            print("(history cleared)", file=sys.stderr)
            continue
        if user.startswith("/image"):
            if not runner.supports_images:
                print("(this model does not support images)", file=sys.stderr)
                continue
            path = Path(user[len("/image"):].strip()).expanduser()
            if not path.name:
                print("usage: /image <path-to-image>", file=sys.stderr)
            elif not path.is_file():
                print(f"(no such file: {path})", file=sys.stderr)
            else:
                staged.append(str(path))
                print(f"(attached {path.name} — will be sent with your next message)",
                      file=sys.stderr)
            continue
        history.append({"role": "user", "content": user})
        print()
        ask(runner, history, staged, args.max_tokens, tools, thinking=thinking)
        staged = []


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit(130)
