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


Q = '<|"|>'


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


def test_gemma4_several_calls_then_a_broken_one_ends_the_search():
    text = ('<|tool_call>call:a{x:1}<tool_call|> <|tool_call>call:b{}<tool_call|>'
            '<|tool_call>call:c{y:<|"|>open<tool_call|><|tool_call>call:d{}<tool_call|>')
    content, calls = GEMMA4.parse(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert content == '<|tool_call>call:c{y:<|"|>open<tool_call|><|tool_call>call:d{}<tool_call|>'


def test_gemma4_a_call_quoted_inside_an_argument_is_not_a_call():
    inner = "<|tool_call>call:rm{}<tool_call|>"
    content, calls = GEMMA4.parse(f'<|tool_call>call:write{{body:{Q}see {inner}{Q}}}<tool_call|>')
    assert content == "" and args_of(calls) == [("write", {"body": f"see {inner}"})]
    # Quotes inside the copied text end the string early: the outer call is
    # broken, and the search stops there instead of finding the inner one.
    for text in (f'<|tool_call>call:write{{body:{Q}<|tool_call>call:rm{{path:{Q}/{Q}}}<tool_call|>{Q}}}<tool_call|>',
                 f'<|tool_call>call:write{{body:{Q}unclosed {inner}'):
        assert GEMMA4.parse(text) == (text, None)


def test_gemma4_tolerates_whitespace_and_rejects_what_it_cannot_read():
    spaced = f'<|tool_call> call:get_weather {{ city : {Q}Lomé{Q} ,\n days: 3 , tags: [ {Q}a{Q} , 2 ] }} <tool_call|>'
    assert args_of(GEMMA4.parse(spaced)[1]) == [
        ("get_weather", {"city": "Lomé", "days": 3, "tags": ["a", 2]})]
    assert args_of(GEMMA4.parse('<|tool_call>call:f{time:10:30,url:https://x.io/a,n:None}<tool_call|>')[1]) == [
        ("f", {"time": "10:30", "url": "https://x.io/a", "n": None})]
    assert args_of(GEMMA4.parse('<|tool_call>call:f{"a":"b","n":3}<tool_call|>')[1]) == [
        ("f", {"a": "b", "n": 3})]
    for broken in ("{a:1,,b:2}", "{a:,b:1}", "{:1}", "{a 1}", f"{{a:{Q}x{Q} b:2}}", f"{{a:{Q}x{Q}{Q}y{Q}}}",
                   "{a:<|other|>}", "{a:1}}"):
        text = f"<|tool_call>call:f{broken}<tool_call|>"
        assert GEMMA4.parse(text) == (text, None), broken


def test_a_marker_mentioned_in_prose_does_not_hide_the_call_after_it():
    content, calls = GEMMA4.parse('The <|tool_call> token starts a call. <|tool_call>call:f{}<tool_call|>')
    assert content == "The <|tool_call> token starts a call." and args_of(calls) == [("f", {})]
    content, calls = HERMES.parse('Wrap it in <tool_call> tags.\n<tool_call>{"name": "f", "arguments": {}}</tool_call>')
    assert content == "Wrap it in <tool_call> tags." and args_of(calls) == [("f", {})]


@pytest.mark.parametrize("fmt,text", [
    (GEMMA4, "<|tool_call>call:f{a:" + "[" * 5000 + "}<tool_call|>"),
    (GEMMA4, "<|tool_call>call:f" + "{a:" * 5000 + "}<tool_call|>"),
    (HERMES, '<tool_call>{"name":"f","arguments":{"a":' + "[" * 5000),
    (QWEN_XML, "<tool_call><function=f><parameter=a>" + "[" * 100000 + "</parameter></function></tool_call>"),
])
def test_runaway_nesting_is_text_not_a_crash(fmt, text):
    content, calls = fmt.parse(text)
    assert calls is None or args_of(calls) == [("f", {"a": "[" * 100000})]
    assert content in (text, "")


@pytest.mark.parametrize("fmt,text", [
    (GEMMA4, "<|tool_call>call:f{a:NaN,b:Infinity}<tool_call|>"),
    (HERMES, '<tool_call>{"name":"f","arguments":{"a":NaN}}</tool_call>'),
])
def test_arguments_are_always_valid_json(fmt, text):
    _, calls = fmt.parse(text)
    for call in calls or []:
        json.loads(call["function"]["arguments"], parse_constant=lambda name: pytest.fail(name))


def test_only_qwen_strips_a_leading_think_block():
    assert QWEN_XML.parse("<think>hmm</think>\n\nAnswer.")[0] == "Answer."
    hermes = '<tool_call>{"name": "w", "arguments": {"c": "close </think> ok"}}</tool_call>'
    assert args_of(HERMES.parse(hermes)[1]) == [("w", {"c": "close </think> ok"})]
    gemma = f'<|tool_call>call:w{{c:{Q}a </think> b{Q}}}<tool_call|>'
    assert args_of(GEMMA4.parse(gemma)[1]) == [("w", {"c": "a </think> b"})]


def test_gemma4_history_drops_text_that_came_before_a_call():
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Checking.", "tool_calls": [{"id": "c1"}]},
                {"role": "tool", "content": "sunny"},
                {"role": "assistant", "content": "It is sunny."}]
    out = GEMMA4.prepare_history(messages)
    assert [m["content"] for m in out] == ["hi", "", "sunny", "It is sunny."]
    assert messages[1]["content"] == "Checking.", "the caller's messages are not edited"
    assert HERMES.prepare_history(messages) is messages and QWEN_XML.prepare_history(messages) is messages


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


def test_hermes_string_arguments_and_parameters():
    text = ('<tool_call>{"name": "a", "arguments": "{\\"q\\": \\"1.0\\"}"}</tool_call>'
            'mid<tool_call>{"name": "b", "parameters": {"z": [1, "2"]}}</tool_call>')
    content, calls = HERMES.parse(text)
    assert args_of(calls) == [("a", {"q": "1.0"}), ("b", {"z": [1, "2"]})]
    assert content == "mid"


def test_hermes_calls_without_closing_tags_and_text_after_them():
    text = ('<tool_call>\n{{"name": "a", "arguments": {"x": 1}}\n'
            '<tool_call>\n{{"name": "b", "arguments": {}}}\nDone, see above.')
    content, calls = HERMES.parse(text)
    assert args_of(calls) == [("a", {"x": 1}), ("b", {})]
    assert content == "Done, see above."


def test_hermes_closing_tag_inside_an_argument_is_part_of_the_string():
    text = '<tool_call>{"name": "edit", "arguments": {"old": "x = \\"</tool_call>\\"", "new": "<tool_call>"}}</tool_call>'
    content, calls = HERMES.parse(text)
    assert content == ""
    assert args_of(calls) == [("edit", {"old": 'x = "</tool_call>"', "new": "<tool_call>"})]


def test_hermes_a_broken_call_is_text_and_ends_the_search():
    for body in ("{not json}", '{"name": 3, "arguments": {}}', '{"name": "f", "arguments": [1]}',
                 '{"name": "f", "arguments": "nope"}'):
        text = f'<tool_call>{body}</tool_call><tool_call>{{"name": "g", "arguments": {{}}}}</tool_call>'
        assert HERMES.parse(text) == (text, None), body


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
