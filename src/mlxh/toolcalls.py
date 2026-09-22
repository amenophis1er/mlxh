"""Tool-call parsing (Qwen-style XML) and user-tool loading.

mlxh ships no tools of its own. `chat --tools` loads them from
$MLXH_HOME/tools.py (default ~/.mlxh/tools.py), a user-owned module defining:

    TOOL_REGISTRY = {"tool_name": callable(**kwargs) -> dict}
    TOOL_SPECS    = [ ... OpenAI function-tool specs ... ]

A ready-made example (weather, time, calculator) ships in the repo at
examples/tools.py — copy it there to try tools out.

The parsing here is core, not demo: the API server uses it to translate a
model's XML tool calls into OpenAI `tool_calls` responses.
"""

import importlib.util
import json
import os
import re
import uuid
from pathlib import Path

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S
)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text):
    """Split model output into (content, openai_tool_calls_or_None)."""
    text = re.sub(r"^\s*(<think>)?.*?</think>\s*", "", text, count=1, flags=re.S)
    calls = []
    for name, body in TOOL_CALL_RE.findall(text):
        args = {}
        for key, value in PARAM_RE.findall(body):
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    content = TOOL_CALL_RE.sub("", text).strip()
    return content, calls or None


def load_user_tools():
    """Load TOOL_REGISTRY/TOOL_SPECS from the user's tools file.

    Returns (registry, specs, path); registry is None when the file
    doesn't exist.
    """
    path = Path(os.environ.get("MLXH_HOME", Path.home() / ".mlxh")) / "tools.py"
    if not path.is_file():
        return None, None, path
    spec = importlib.util.spec_from_file_location("mlxh_user_tools", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "TOOL_REGISTRY", {}), getattr(mod, "TOOL_SPECS", []), path


def run_tool(registry, name, args):
    fn = registry.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}
    except Exception as e:
        return {"error": str(e)}
