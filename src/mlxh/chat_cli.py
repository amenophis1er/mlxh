"""Terminal chat for any mlxh-managed model, with built-in tools.

Invoked by `mlxh chat <model>`; also standalone:
    python chat_cli.py --model-path /path/to/model [-p "question"] [-i img.png]
"""

import argparse
import atexit
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from . import ui
from .loader import load_runner
from .toolcalls import load_user_tools, parse_tool_calls, run_tool

MAX_TOOL_ROUNDS = 5
MAX_IMAGE_DOWNLOAD = 25 * 1024 * 1024


def _validate_image(path):
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(f"not a readable image: {exc}") from None


def _download_image(url):
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "mlxh/0.1"})
    try:
        response = urllib.request.urlopen(request, timeout=15)
    except Exception as exc:
        raise ValueError(f"could not download image: {exc}") from None
    path = None
    try:
        final_url = response.geturl()
        if urlparse(final_url).scheme not in ("http", "https"):
            raise ValueError("image redirects must stay on HTTP or HTTPS")
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
        if not content_type.startswith("image/"):
            raise ValueError(f"URL returned {content_type or 'unknown content type'}, not an image")
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_IMAGE_DOWNLOAD:
            raise ValueError("image is larger than the 25 MB download limit")
        suffix = {
            "image/gif": ".gif", "image/jpeg": ".jpg", "image/png": ".png",
            "image/tiff": ".tiff", "image/webp": ".webp",
        }.get(content_type, ".img")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as file:
            path = Path(file.name)
            total = 0
            while chunk := response.read(64 * 1024):
                total += len(chunk)
                if total > MAX_IMAGE_DOWNLOAD:
                    raise ValueError("image is larger than the 25 MB download limit")
                file.write(chunk)
        _validate_image(path)
        return str(path)
    except Exception:
        if path:
            path.unlink(missing_ok=True)
        raise
    finally:
        response.close()


def _clipboard_image():
    import subprocess

    attempts = (("«class PNGf»", ".png"), ("TIFF picture", ".tiff"))
    last_error = "clipboard does not contain an image"
    for clipboard_type, suffix in attempts:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as file:
            path = Path(file.name)
        script = f"""
on run argv
    set imageData to the clipboard as {clipboard_type}
    set outputFile to open for access POSIX file (item 1 of argv) with write permission
    try
        set eof outputFile to 0
        write imageData to outputFile
        close access outputFile
    on error message
        try
            close access outputFile
        end try
        error message
    end try
end run
"""
        try:
            result = subprocess.run(
                ["osascript", "-e", script, str(path)],
                capture_output=True, text=True, check=False,
            )
        except OSError as exc:
            path.unlink(missing_ok=True)
            raise ValueError(f"could not read the clipboard: {exc}") from None
        if result.returncode == 0:
            try:
                _validate_image(path)
                return str(path)
            except ValueError as exc:
                last_error = str(exc)
        else:
            last_error = result.stderr.strip() or last_error
        path.unlink(missing_ok=True)
    raise ValueError(f"could not read an image from the clipboard: {last_error}")


def _prepare_image(source=None):
    """Return (local path, is_temporary) for clipboard, URL, or file input."""
    if not source:
        return _clipboard_image(), True
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return _download_image(source), True
    if parsed.scheme:
        raise ValueError("image URL must use HTTP or HTTPS")
    try:
        parts = shlex.split(source)
        if len(parts) == 1:
            source = parts[0]
    except ValueError:
        pass
    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError(f"no such file: {path}")
    _validate_image(path)
    return str(path), False


def _cleanup_images(paths):
    for path in tuple(paths):
        Path(path).unlink(missing_ok=True)
        paths.discard(path)


def _chat_session(history_path, commands, input=None, output=None):
    """Build the interactive editor; bracketed pastes stay one editable input."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.history import FileHistory

    class CompatibleFileHistory(FileHistory):
        """Read legacy readline lines plus prompt_toolkit multiline entries."""

        def load_history_strings(self):
            strings, entry = [], []
            if Path(self.filename).is_file():
                for line in Path(self.filename).read_text(errors="replace").splitlines():
                    if line.startswith("+"):
                        entry.append(line[1:])
                    else:
                        if entry:
                            strings.append("\n".join(entry))
                            entry = []
                        if line and not line.startswith("# "):
                            strings.append(line)
                if entry:
                    strings.append("\n".join(entry))
            return reversed(strings[-500:])

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/") or " " in text:
                return
            for command in commands:
                if command.startswith(text):
                    yield Completion(command, start_position=-len(text))

    return PromptSession(
        history=CompatibleFileHistory(str(history_path)),
        completer=SlashCompleter(),
        complete_while_typing=False,
        input=input,
        output=output,
    )


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
    ap.add_argument("-i", "--image", action="append", default=[],
                    help="local image path or HTTP(S) image URL to include")
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

    temporary_images = set()
    atexit.register(_cleanup_images, temporary_images)

    if args.prompt:
        images = []
        try:
            for source in args.image:
                path, temporary = _prepare_image(source)
                images.append(path)
                if temporary:
                    temporary_images.add(path)
        except ValueError as exc:
            _cleanup_images(temporary_images)
            sys.exit(f"--image: {exc}")
        messages = [{"role": "user", "content": args.prompt}]
        try:
            ask(runner, messages, images, args.max_tokens, tools, thinking=thinking)
        finally:
            _cleanup_images(temporary_images)
        return

    hist = Path(os.environ.get("MLXH_HOME", Path.home() / ".mlxh")) / "chat_history"
    hist.parent.mkdir(parents=True, exist_ok=True)
    cmds = ["/exit", "/reset", "/help", "/bye", "/quit"]
    if runner.supports_images:
        cmds.insert(0, "/image ")
    session = _chat_session(hist, cmds)
    prompt_ansi = (sys.stdin.isatty() and sys.stdout.isatty()
                   and not os.environ.get("NO_COLOR"))

    hint = "/image [path|URL] attaches an image (no argument: clipboard), " \
        if runner.supports_images else ""
    print(f"Interactive chat. {hint}/reset clears history, /exit quits, /help lists commands.",
          file=sys.stderr)
    history, staged = [], []
    while True:
        tag = f" [{len(staged)} img]" if staged else ""
        if prompt_ansi:
            prompt = [("", "\n"), ("bold ansicyan", f"you{tag}> ")]
        else:
            prompt = f"\nyou{tag}> "
        try:
            user = session.prompt(prompt, prompt_continuation="... ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/bye", "/quit"):
            break
        if user == "/help":
            image_help = (
                "/image [path|URL]  attach a file, URL, or clipboard image\n"
                if runner.supports_images else ""
            )
            print(image_help +
                  "/reset         clear conversation history\n"
                  "/exit          quit (also /bye, /quit, Ctrl-D)", file=sys.stderr)
            continue
        if user == "/reset":
            _cleanup_images(temporary_images)
            history, staged = [], []
            print("(history cleared)", file=sys.stderr)
            continue
        if user == "/image" or user.startswith("/image "):
            if not runner.supports_images:
                print("(this model does not support images)", file=sys.stderr)
                continue
            source = user[len("/image"):].strip() or None
            try:
                path, temporary = _prepare_image(source)
            except ValueError as exc:
                print(f"({exc})", file=sys.stderr)
                continue
            staged.append(path)
            if temporary:
                temporary_images.add(path)
            label = "clipboard image" if source is None else source
            print(f"(attached {label} — will be sent with your next message)",
                  file=sys.stderr)
            continue
        history.append({"role": "user", "content": user})
        print()
        try:
            ask(runner, history, staged, args.max_tokens, tools, thinking=thinking)
        finally:
            _cleanup_images(temporary_images)
        staged = []

    _cleanup_images(temporary_images)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        sys.exit(130)
