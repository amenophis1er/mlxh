import json

from mlxh import responses_compat as oresp


def test_string_input_and_instructions():
    oai = oresp.to_openai_body({"instructions": "be brief", "input": "hi",
                                "max_output_tokens": 50})
    assert oai["messages"][0] == {"role": "system", "content": "be brief"}
    assert oai["messages"][1] == {"role": "user", "content": "hi"}
    assert oai["max_tokens"] == 50


def test_message_items_and_second_system_demoted():
    body = {"instructions": "sys", "input": [
        {"type": "message", "role": "system",
         "content": [{"type": "input_text", "text": "reminder"}]},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "q"},
                     {"type": "input_image", "image_url": "data:image/png;base64,QQ=="}]},
    ]}
    msgs = oresp.to_openai_body(body)["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "user"]
    parts = msgs[2]["content"]
    assert parts[0] == {"type": "text", "text": "q"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png")


def test_function_call_round_trip():
    body = {"tools": [{"type": "function", "name": "get_weather",
                       "description": "w", "parameters": {"type": "object"}}],
            "input": [
                {"type": "message", "role": "user", "content": "weather?"},
                {"type": "function_call", "call_id": "call_1",
                 "name": "get_weather", "arguments": '{"city": "Paris"}'},
                {"type": "function_call_output", "call_id": "call_1",
                 "output": "22C"},
            ]}
    oai = oresp.to_openai_body(body)
    assert oai["tools"][0]["function"]["name"] == "get_weather"
    assert oai["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'
    assert oai["messages"][2] == {"role": "tool", "content": "22C"}


def test_output_items_and_response_object():
    calls = [{"id": "call_a", "type": "function",
              "function": {"name": "f", "arguments": '{"x":1}'}}]
    items = oresp.output_items("hello", calls)
    assert items[0]["type"] == "message"
    assert items[0]["content"][0]["text"] == "hello"
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_a" and items[1]["name"] == "f"
    resp = oresp.response_object("resp_1", "m", items, oresp.usage_of(10, 5))
    assert resp["status"] == "completed"
    assert resp["usage"]["total_tokens"] == 15
