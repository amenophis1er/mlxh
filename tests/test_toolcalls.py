import importlib.util
import json
from pathlib import Path

import pytest

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


# --- per-family formats ---------------------------------------------------

from mlxh import toolcalls
from mlxh.toolcalls import GEMMA4, HERMES, QWEN_XML


def args_of(calls):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]


def test_gemma4_call_as_the_model_writes_it():
    text = '<|tool_call>call:get_weather{city:<|"|>Lomé<|"|>,days:3}<tool_call|>'
    content, calls = GEMMA4.parse(text)
    assert content == ""
    assert args_of(calls) == [("get_weather", {"city": "Lomé", "days": 3})]


def test_gemma4_values_round_trip_exactly():
    text = ('Let me look.<|tool_call>call:f{zip:<|"|>01234<|"|>,n:<|"|>3<|"|>,ok:true,'
            'none:null,x:-2.5,tags:[<|"|>a,b<|"|>,7],deep:{k:<|"|>{v}: w<|"|>,l:[]},'
            'note:<|"|>say "hi"<|"|>}<tool_call|>')
    content, calls = GEMMA4.parse(text)
    assert content == "Let me look."
    assert args_of(calls)[0][1] == {
        "zip": "01234", "n": "3", "ok": True, "none": None, "x": -2.5,
        "tags": ["a,b", 7], "deep": {"k": "{v}: w", "l": []}, "note": 'say "hi"'}


def test_gemma4_several_calls_and_a_broken_one_kept_as_text():
    text = ('<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:<|"|>open'
            '<tool_call|><|tool_call>call:c{}<tool_call|>')
    content, calls = GEMMA4.parse(text)
    assert [c["function"]["name"] for c in calls] == ["a", "c"]
    assert content == '<|tool_call>call:b{y:<|"|>open<tool_call|>'


def test_hermes_call():
    text = 'Checking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Lomé", "days": "3"}}\n</tool_call>'
    content, calls = HERMES.parse(text)
    assert content == "Checking."
    assert args_of(calls) == [("get_weather", {"city": "Lomé", "days": "3"})]


def test_hermes_as_qwen25_writes_it_doubled_brace_and_no_closing_tag():
    text = '<tool_call>\n{{"name": "get_weather", "arguments": {"city": "Lomé", "days": 3}}'
    content, calls = HERMES.parse(text)
    assert content == ""
    assert args_of(calls) == [("get_weather", {"city": "Lomé", "days": 3})]


def test_hermes_string_arguments_several_calls_and_junk():
    text = ('<tool_call>{"name": "a", "arguments": "{\\"q\\": \\"1.0\\"}"}</tool_call>'
            'mid<tool_call>not json</tool_call>'
            '<tool_call>{"name": "b", "parameters": {"z": [1, "2"]}}</tool_call>')
    content, calls = HERMES.parse(text)
    assert args_of(calls) == [("a", {"q": "1.0"}), ("b", {"z": [1, "2"]})]
    assert content == "mid<tool_call>not json</tool_call>"


def test_qwen_xml_is_unchanged_and_the_default():
    content, calls = toolcalls.parse_tool_calls(XML)
    assert (content, args_of(calls)) == ("I'll check.", args_of(QWEN_XML.parse(XML)[1]))
    assert toolcalls.format_for_template("") is QWEN_XML


def test_markers_per_format():
    assert (QWEN_XML.marker, HERMES.marker, GEMMA4.marker) == ("<tool_call>", "<tool_call>", "<|tool_call>")


@pytest.mark.parametrize("template,fmt", [
    ("{{- '<|tool_call>call:' + function['name'] + '{' -}}", GEMMA4),
    ("{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}", QWEN_XML),
    ('<tool_call>\\n{{"name": <function-name>, "arguments": <args-json-object>}}\\n</tool_call>', HERMES),
    ("{{ messages }}", QWEN_XML),
])
def test_format_follows_the_chat_template(template, fmt):
    assert toolcalls.format_for_template(template) is fmt


def test_detect_reads_jinja_tokenizer_config_and_named_templates(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": [{"name": "default", "template": "<tool_call>{json}</tool_call>"}]}))
    assert toolcalls.detect_format(tmp_path) is HERMES
    (tmp_path / "chat_template.jinja").write_text("<|tool_call>call:")
    assert toolcalls.detect_format(tmp_path) is GEMMA4
    assert toolcalls.detect_format(tmp_path / "missing") is QWEN_XML
