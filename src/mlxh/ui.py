"""Terminal output helpers: short status lines, color only when it's a TTY."""

import os
import sys

from rich.markdown import ListItem, Markdown
from rich.segment import Segment


def _colors_on(stream):
    return stream.isatty() and not os.environ.get("NO_COLOR")


def _c(code, s, stream):
    return f"\033[{code}m{s}\033[0m" if _colors_on(stream) else s


def bold(s, stream=sys.stdout):
    return _c("1", s, stream)


def dim(s, stream=sys.stdout):
    return _c("2", s, stream)


def ok(msg):
    print(_c("32", "✓", sys.stdout) + " " + msg)


def step(msg):
    print(_c("36", "→", sys.stdout) + " " + msg)


def note(msg):
    print(dim("  " + msg))


def fail(msg, *details, hint=None):
    """Print an error block to stderr and exit 1."""
    print(_c("31", "✗", sys.stderr) + " " + msg, file=sys.stderr)
    for d in details:
        print("  " + d, file=sys.stderr)
    if hint:
        print("  " + dim("→ " + hint, sys.stderr), file=sys.stderr)
    sys.exit(1)


def select(title, options, annotations=None):
    """Arrow-key selector on a TTY. Returns the chosen index.

    Up/Down or j/k moves, Enter confirms, 1-9 jumps and confirms,
    q/Esc/Ctrl-C cancels (exit 130). Falls back to a numbered prompt
    when raw terminal mode isn't available.
    """
    annotations = annotations or [""] * len(options)
    width = max(len(o) for o in options)

    def line(i, selected):
        mark = _c("36", "❯", sys.stdout) if selected else " "
        padded = options[i].ljust(width)
        name = bold(padded) if selected else padded
        note = ("  " + dim(annotations[i])) if annotations[i] else ""
        return f"{mark} {name}{note}"

    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    except Exception:
        return _select_numbered(title, options)

    import select as _select
    idx = 0
    print(dim(title))
    sys.stdout.write("\x1b[?25l")  # hide cursor
    try:
        for i in range(len(options)):
            print(line(i, i == idx))
        tty.setcbreak(fd)
        while True:
            # os.read on the fd, not sys.stdin.read: Python's buffering would
            # swallow the rest of an escape sequence and select() would then
            # misread an arrow key as a bare Esc.
            try:
                ch = os.read(fd, 1).decode(errors="replace")
            except KeyboardInterrupt:
                sys.exit(130)
            if ch == "\x1b":
                if _select.select([fd], [], [], 0.05)[0]:
                    seq = os.read(fd, 2).decode(errors="replace")
                    if seq == "[A":
                        idx = (idx - 1) % len(options)
                    elif seq == "[B":
                        idx = (idx + 1) % len(options)
                else:  # bare Esc
                    sys.exit(130)
            elif ch in ("k",):
                idx = (idx - 1) % len(options)
            elif ch in ("j",):
                idx = (idx + 1) % len(options)
            elif ch.isdigit() and 1 <= int(ch) <= len(options):
                idx = int(ch) - 1
                break
            elif ch in ("\r", "\n"):
                break
            elif ch in ("q", "\x03", "\x04"):
                sys.exit(130)
            sys.stdout.write(f"\x1b[{len(options)}A")
            for i in range(len(options)):
                sys.stdout.write("\x1b[2K" + line(i, i == idx) + "\n")
            sys.stdout.flush()
        return idx
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()


def _select_numbered(title, options):
    print(dim(title))
    for i, name in enumerate(options, 1):
        print(f"  {i}) {name}")
    while True:
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(130)
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice) - 1
        if choice in options:
            return options.index(choice)
        print(dim(f"enter 1-{len(options)} or a name"))


class _DottedListItem(ListItem):
    """Rich list item that keeps the conventional period after its number."""

    def render_number(self, console, options, number, last_number):
        number_width = len(str(last_number)) + 3
        render_options = options.update(width=options.max_width - number_width)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        number_style = console.get_style("markdown.item.number", default="none")
        padding = Segment(" " * number_width, number_style)
        numeral = Segment(
            f"{number}.".rjust(number_width - 1) + " ", number_style
        )
        for index, line in enumerate(lines):
            yield numeral if index == 0 else padding
            yield from line
            yield Segment.line()


class _ChatMarkdown(Markdown):
    elements = {**Markdown.elements, "list_item_open": _DottedListItem}


class StreamRenderer:
    """Live-render streamed Markdown, with plain pass-through when piped."""

    def __init__(self, out=None, think_open=False, force_terminal=None):
        self.out = out or sys.stdout
        self.enabled = (_colors_on(self.out) if force_terminal is None
                        else force_terminal)
        self.reasoning = think_open
        self.reasoning_text = ""
        self.markdown_text = ""
        self.pending_markdown = ""
        self._needs_block_gap = False
        self._live = None
        self._console = None

    def _rich(self):
        if self._console is None:
            from rich.console import Console
            self._console = Console(
                file=self.out, force_terminal=True, color_system="auto",
                highlight=False,
            )
        return self._console

    def _update(self, renderable):
        from rich.live import Live
        if self._live is None:
            self._live = Live(
                renderable, console=self._rich(), auto_refresh=False,
                transient=False, vertical_overflow="crop",
            )
            self._live.start(refresh=True)
        else:
            self._live.update(renderable, refresh=True)

    def _stop_live(self):
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _render_reasoning(self, text):
        from rich.text import Text
        self._update(Text(text, style="dim"))

    def _render_markdown(self, text):
        self._update(_ChatMarkdown(text, code_theme="monokai"))

    def _start_markdown_block(self, text):
        stripped = text.lstrip()
        heading_like = stripped.startswith("#") or stripped.startswith("**")
        if self._needs_block_gap and heading_like:
            self._rich().print()
        self._needs_block_gap = False

    @staticmethod
    def _block_boundary(text):
        """Last blank-line boundary outside a fenced code block."""
        fenced = False
        offset = boundary = 0
        for line in text.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fenced = not fenced
            offset += len(line)
            if not fenced and not line.strip() and line.endswith("\n"):
                boundary = offset
        return boundary

    def feed(self, text):
        if not self.enabled:
            self.out.write(text)
            self.out.flush()
            return
        if self.reasoning:
            self.reasoning_text += text
            tag = "</think>"
            cut = self.reasoning_text.find(tag)
            if cut == -1:
                safe = max(0, len(self.reasoning_text) - len(tag))
                self._render_reasoning(self.reasoning_text[:safe])
                return
            self._render_reasoning(self.reasoning_text[:cut])
            self._stop_live()
            self._rich().print()
            text = self.reasoning_text[cut + len(tag):].lstrip("\n")
            self.reasoning_text, self.reasoning = "", False
            if not text:
                return
        self.markdown_text += text
        self.pending_markdown += text
        boundary = self._block_boundary(self.pending_markdown)
        if boundary:
            complete = self.pending_markdown[:boundary]
            self._start_markdown_block(complete)
            self._render_markdown(complete)
            self._stop_live()
            self._needs_block_gap = True
            self.pending_markdown = self.pending_markdown[boundary:]
        if self.pending_markdown:
            self._start_markdown_block(self.pending_markdown)
            self._render_markdown(self.pending_markdown)

    def finish(self):
        if not self.enabled:
            return
        if self.reasoning:
            self._render_reasoning(self.reasoning_text)
            self.reasoning_text, self.reasoning = "", False
        elif self.pending_markdown:
            self._render_markdown(self.pending_markdown)
        self._stop_live()
        self.pending_markdown = ""
        self.out.flush()
