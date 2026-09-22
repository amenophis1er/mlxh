"""Terminal chat for any mlxh-managed model, with built-in tools.

Invoked by `mlxh chat <model>`; also standalone:
    python chat_cli.py --model-path /path/to/model [-p "question"] [-i img.png]
"""

import argparse
import json
import sys
from pathlib import Path

from .loader import load_runner
from .toolcalls import TOOL_REGISTRY, TOOL_SPECS, parse_tool_calls, run_tool

MAX_TOOL_ROUNDS = 5


def generate_once(runner, messages, images, max_tokens, tools):
    """One generation pass; streams visible text, hides tool-call XML."""
    prompt = runner.template(
        messages, num_images=len(images), tools=TOOL_SPECS if tools else None
    )
    parts, printed = [], 0
    marker = "<tool_call>"
    last = None
    for resp in runner.stream(prompt, images=images, max_tokens=max_tokens):
        parts.append(resp.text)
        full = "".join(parts)
        cut = full.find(marker)
        # hold back a tag-length tail so a half-arrived "<tool_ca" never prints
        visible = full[:cut] if cut != -1 else full[: max(0, len(full) - len(marker))]
        if len(visible) > printed:
            print(visible[printed:], end="", flush=True)
            printed = len(visible)
        last = resp
    full = "".join(parts)
    cut = full.find(marker)
    visible = full[:cut] if cut != -1 else full
    if len(visible) > printed:
        print(visible[printed:], end="", flush=True)
    return full, last


def ask(runner, messages, images, max_tokens, tools=True):
    total_tokens, tps = 0, 0.0
    for _ in range(MAX_TOOL_ROUNDS):
        text, last = generate_once(runner, messages, images, max_tokens, tools)
        total_tokens += last.generation_tokens
        tps = last.generation_tps
        content, tool_calls = parse_tool_calls(text)
        if not tool_calls:
            print(f"\n\n[{total_tokens} tokens @ {tps:.1f} tok/s]")
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
            result = run_tool(name, args)
            print(f"\n[tool] {name}({json.dumps(args, ensure_ascii=False)}) -> "
                  f"{json.dumps(result, ensure_ascii=False)}", file=sys.stderr)
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
    ap.add_argument("--no-tools", action="store_true", help="disable built-in tools")
    args = ap.parse_args()

    print(f"Loading {Path(args.model_path).name}...", file=sys.stderr)
    runner = load_runner(args.model_path)
    if args.image and not runner.supports_images:
        sys.exit("this model does not support images")
    tools = not args.no_tools
    if tools:
        print(f"Tools enabled: {', '.join(TOOL_REGISTRY)}", file=sys.stderr)

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        ask(runner, messages, args.image, args.max_tokens, tools)
        return

    hint = "/image <path> attaches an image to your next message,\n" if runner.supports_images else ""
    print(f"Interactive chat. {hint}/reset clears history, Ctrl-D exits.", file=sys.stderr)
    history, staged = [], []
    while True:
        tag = f" [{len(staged)} img]" if staged else ""
        try:
            user = input(f"\nyou{tag}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
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
        ask(runner, history, staged, args.max_tokens, tools)
        staged = []


if __name__ == "__main__":
    main()
