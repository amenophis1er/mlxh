import subprocess
import sys

from mlxh.worker_supervisor import supervisor_argv


def test_worker_supervisor_stops_child_when_manager_is_gone():
    process = subprocess.Popen(
        supervisor_argv([sys.executable, "-c", "import time; time.sleep(60)"],
                        2**30),
        start_new_session=True,
    )
    try:
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode is not None
