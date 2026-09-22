import json

from mlxh import anthropic_compat as anth


def test_system_and_text():
    body = {"system": "be brief", "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}]}
    oai = anth.to_openai_body(body)
    assert oai["messages"][0] == {"role": "system", "content": "be brief"}
    assert oai["messages"][1] == {"role": "user", "content": "hi"}
    assert oai["max_tokens"] == 50


def test_system_blocks_and_sampling():
    body = {"system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "temperature": 0.2, "messages": []}
    oai = anth.to_openai_body(body)
    assert oai["messages"][0]["content"] == "a\nb"
    assert oai["temperature"] == 0.2
    assert oai["max_tokens"] == 1024  # default


def test_image_block():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
    ]}]}
    parts = anth.to_openai_body(body)["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


def test_tool_use_and_result_round_trip():
    body = {
        "tools": [{"name": "get_weather", "description": "w",
                   "input_schema": {"type": "object", "properties": {}}}],
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Paris"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "22C"}]}]},
        ],
    }
    oai = anth.to_openai_body(body)
    assert oai["tools"][0]["function"]["name"] == "get_weather"
    assistant = oai["messages"][1]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
    assert oai["messages"][2] == {"role": "tool", "content": "22C"}


def test_message_response_with_tool_call():
    calls = [{"id": "call_abc", "type": "function",
              "function": {"name": "f", "arguments": '{"x": 1}'}}]
    resp = anth.message_response("m", "thinking", calls, 10, 5, None)
    assert resp["stop_reason"] == "tool_use"
    assert resp["content"][0] == {"type": "text", "text": "thinking"}
    tool = resp["content"][1]
    assert tool["type"] == "tool_use" and tool["id"].startswith("toolu_")
    assert tool["input"] == {"x": 1}
    assert resp["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_stop_reasons():
    assert anth.stop_reason(None, "stop") == "end_turn"
    assert anth.stop_reason(None, "length") == "max_tokens"
    assert anth.stop_reason([{"any": 1}], "stop") == "tool_use"


def test_sse_event_format():
    ev = anth.sse_event("message_stop", {"type": "message_stop"})
    assert ev == 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


def test_mid_conversation_system_demoted_to_user():
    body = {"system": "top", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "reminder"},
    ]}
    msgs = anth.to_openai_body(body)["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "user"]
    assert msgs[2]["content"] == "reminder"
