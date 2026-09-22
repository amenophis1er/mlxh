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

    def line(i, selected):
        mark = _c("36", "❯", sys.stdout) if selected else " "
        name = bold(options[i]) if selected else options[i]
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
            ch = os.read(fd, 1).decode(errors="replace")
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
