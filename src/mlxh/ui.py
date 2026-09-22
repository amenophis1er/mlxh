"""Terminal output helpers: short status lines, color only when it's a TTY."""

import os
import sys


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


class StreamRenderer:
    """Stream model output with lightweight markdown styling.

    Handles **bold**, `inline code` (cyan), ``` fenced blocks (dim), and
    # headings (bold) — statefully, so markers split across chunks still
    work. Plain pass-through when the stream isn't a color TTY.
    """

    BOLD, UNBOLD = "\x1b[1m", "\x1b[22m"
    CODE, UNCODE = "\x1b[36m", "\x1b[39m"
    DIM, UNDIM = "\x1b[2m", "\x1b[22m"

    def __init__(self, out=None, think_open=False):
        self.out = out or sys.stdout
        self.enabled = _colors_on(self.out)
        self.carry = ""
        self.bold = self.code = self.fence = self.heading = False
        self.line_start = True
        # think_open: the chat template pre-opened a <think> block, so the
        # stream begins with reasoning; render it dim until </think>.
        self.reasoning = think_open
        self._rbuf = ""
        if self.reasoning and self.enabled:
            self.out.write(self.DIM)

    def feed(self, text):
        if not self.enabled:
            self.out.write(text)
            self.out.flush()
            return
        if self.reasoning:
            self._rbuf += text
            tag = "</think>"
            cut = self._rbuf.find(tag)
            if cut == -1:
                safe = max(0, len(self._rbuf) - len(tag))
                self.out.write(self._rbuf[:safe])
                self._rbuf = self._rbuf[safe:]
                self.out.flush()
                return
            self.out.write(self._rbuf[:cut] + self.UNDIM + "\n\n")
            text = self._rbuf[cut + len(tag):].lstrip("\n")
            self._rbuf, self.reasoning = "", False
            if not text:
                return
        buf, self.carry = self.carry + text, ""
        self._process(buf, final=False)

    def finish(self):
        if not self.enabled:
            return
        if self.reasoning:
            self.out.write(self._rbuf + "\x1b[0m")
            self._rbuf, self.reasoning = "", False
            self.out.flush()
            return
        if self.carry:
            buf, self.carry = self.carry, ""
            self._process(buf, final=True)
        if self.bold or self.code or self.fence or self.heading:
            self.out.write("\x1b[0m")
        self.out.flush()

    def _process(self, buf, final):
        w = self.out.write
        i, n = 0, len(buf)
        while i < n:
            ch = buf[i]
            if ch in "*`":
                j = i
                while j < n and buf[j] == ch:
                    j += 1
                run = j - i
                need = 2 if ch == "*" else 3
                if j == n and not final and run < need:
                    self.carry = buf[i:]
                    break
                if ch == "*" and not self.fence:
                    while run >= 2:
                        self.bold = not self.bold
                        w(self.BOLD if self.bold else self.UNBOLD)
                        run -= 2
                    if run:
                        w("*")
                        self.line_start = False
                elif ch == "`" and run >= 3 and self.line_start:
                    if not self.fence:
                        self.fence = True
                        w(self.DIM + "`" * run)
                    else:
                        w("`" * run + self.UNDIM)
                        self.fence = False
                    self.line_start = False
                elif ch == "`" and run == 1 and not self.fence:
                    self.code = not self.code
                    w(self.CODE if self.code else self.UNCODE)
                else:
                    w(ch * run)
                    self.line_start = False
                i = j
                continue
            if ch == "\n":
                if self.heading:
                    w(self.UNBOLD)
                    self.heading = False
                w("\n")
                self.line_start = True
                i += 1
                continue
            if ch == "#" and self.line_start and not self.fence:
                self.heading = True
                w(self.BOLD + "#")
                self.line_start = False
                i += 1
                continue
            w(ch)
            if ch not in " \t":
                self.line_start = False
            i += 1
        self.out.flush()
