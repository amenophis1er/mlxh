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
