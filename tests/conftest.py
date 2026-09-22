import os
import tempfile

# Must happen before any mlxh import: cli.py resolves MLXH_HOME at import time.
os.environ["MLXH_HOME"] = tempfile.mkdtemp(prefix="mlxh-test-home-")
