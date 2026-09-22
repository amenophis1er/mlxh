import json

from mlxh.toolcalls import calculate, parse_tool_calls, run_tool

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


def test_calculate():
    assert calculate("17 * 23")["result"] == 391
    assert calculate("2 ** 10")["result"] == 1024
    assert calculate("-3 + 0.5")["result"] == -2.5


def test_calculate_rejects_code():
    assert "error" in calculate("__import__('os').system('true')")
    assert "error" in calculate("open('/etc/passwd')")
    assert "error" in calculate("'a' * 99")


def test_run_tool_unknown():
    assert "error" in run_tool("nope", {})


def test_run_tool_bad_args():
    assert "error" in run_tool("calculate", {"wrong": 1})
