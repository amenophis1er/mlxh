import importlib.util
import json
from pathlib import Path

from mlxh.toolcalls import load_user_tools, parse_tool_calls, run_tool

XML = """I'll check.

<tool_call>
<function=get_weather>
<parameter=city>
Paris
</parameter>
<parameter=count>
3
</parameter>
</function>
</tool_call>"""

EXAMPLE = Path(__file__).parent.parent / "examples" / "tools.py"


def load_example():
    spec = importlib.util.spec_from_file_location("example_tools", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_single_call():
    content, calls = parse_tool_calls(XML)
    assert content == "I'll check."
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "get_weather"
    assert json.loads(fn["arguments"]) == {"city": "Paris", "count": 3}
    assert calls[0]["id"].startswith("call_")


def test_parse_no_call():
    content, calls = parse_tool_calls("Just an answer.")
    assert content == "Just an answer."
    assert calls is None


def test_think_block_stripped():
    content, calls = parse_tool_calls("<think>hmm</think>\n\nAnswer.")
    assert content == "Answer."
    assert calls is None


def test_parse_multiple_calls():
    two = XML + (
        "\n<tool_call>\n<function=calculate>\n<parameter=expression>\n"
        "1+1\n</parameter>\n</function>\n</tool_call>"
    )
    _, calls = parse_tool_calls(two)
    assert [c["function"]["name"] for c in calls] == ["get_weather", "calculate"]


def test_load_user_tools_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MLXH_HOME", str(tmp_path))
    registry, specs, path = load_user_tools()
    assert registry is None
    assert path == tmp_path / "tools.py"


def test_load_user_tools_present(tmp_path, monkeypatch):
    monkeypatch.setenv("MLXH_HOME", str(tmp_path))
    (tmp_path / "tools.py").write_text(
        "TOOL_REGISTRY = {'f': lambda x=1: {'ok': x}}\n"
        "TOOL_SPECS = [{'type': 'function', 'function': {'name': 'f'}}]\n"
    )
    registry, specs, _ = load_user_tools()
    assert registry["f"](x=2) == {"ok": 2}
    assert specs[0]["function"]["name"] == "f"


def test_run_tool_dispatch():
    registry = {"echo": lambda **kw: kw}
    assert run_tool(registry, "echo", {"a": 1}) == {"a": 1}
    assert "error" in run_tool(registry, "nope", {})
    assert "error" in run_tool({"f": lambda: 1}, "f", {"unexpected": 1})


# The shipped example must keep working — it's the documented template.

def test_example_contract():
    mod = load_example()
    assert set(mod.TOOL_REGISTRY) == {s["function"]["name"] for s in mod.TOOL_SPECS}


def test_example_calculate():
    mod = load_example()
    calc = mod.TOOL_REGISTRY["calculate"]
    assert calc("17 * 23")["result"] == 391
    assert calc("2 ** 10")["result"] == 1024
    assert "error" in calc("__import__('os').system('true')")
    assert "error" in calc("open('/etc/passwd')")


def test_example_time():
    mod = load_example()
    assert "time" in mod.TOOL_REGISTRY["get_current_time"]()
    assert "error" in mod.TOOL_REGISTRY["get_current_time"]("Not/AZone")
