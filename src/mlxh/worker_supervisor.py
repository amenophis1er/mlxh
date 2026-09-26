"""Small watchdog that ties a model worker's lifetime to its manager."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def supervisor_argv(command: list[str], parent_pid: int) -> list[str]:
    return [sys.executable, "-m", "mlxh.worker_supervisor",
            "--parent-pid", str(parent_pid), "--", *command]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a worker command is required")

    child = subprocess.Popen(command)
    received_signal = None

    def forward(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        if child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, forward)

    while child.poll() is None:
        if received_signal is not None:
            try:
                return child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                return child.wait()
        # A changed parent means the manager exited, including SIGKILL. The
        # manager starts this supervisor as a new-session process-group leader;
        # its model server shares that group and is cleaned up with us.
        if os.getppid() != args.parent_pid:
            child.terminate()
            try:
                return child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                return child.wait()
        time.sleep(0.25)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
