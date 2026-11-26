import json
import queue
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from mlxh import anthropic_compat as anth
from mlxh import serve_app
from mlxh.engine import GenerationTerminal, ReasoningParser
from mlxh.inkling import InklingParser

client = TestClient(serve_app.app)
HI = [{"role": "user", "content": "hi"}]
CALL = '<tool_call>\n<function=get_weather>\n<parameter=city>\nLomé\n</parameter>\n</function>\n</tool_call>'


def generated(*raw, parser=None):
    """Queue what the engine yields for these raw model chunks."""
    parser, out = parser or ReasoningParser(preopened=True), queue.Queue()
    for i, chunk in enumerate(raw):
        final = i == len(raw) - 1
        text, thought = parser.feed(chunk, final=final)
        out.put(NS(text=text, reasoning_text=thought, phase_deltas=list(parser.events),
                   prompt_tokens=5, generation_tokens=i + 1,
                   finish_reason="stop" if final else None))
    out.put(GenerationTerminal("stop", "completed", 5, len(raw), 1.0, 1.0))
    out.put(None)
    return out


ANSWER = ("Count ", "the words.</think>\n", "Hello ", "there.")
TOOL = ("Need weather.</think>", "Checking. ", CALL)


@pytest.fixture
def model(monkeypatch):
    def use(*raw):
        monkeypatch.setattr(serve_app, "submit", lambda *_a: generated(*raw))
    return use


def sse(response):
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: {")]


def test_chat_reasoning_stays_out_of_content(model):
    model(*ANSWER)
    message = client.post("/v1/chat/completions", json={"messages": HI}).json()["choices"][0]["message"]
    assert message["content"] == "Hello there."
    assert message["reasoning_content"] == "Count the words."


def test_chat_without_reasoning_has_no_reasoning_field(model):
    model("</think>Hi.")
    message = client.post("/v1/chat/completions", json={"messages": HI}).json()["choices"][0]["message"]
    assert "reasoning_content" not in message


@pytest.mark.parametrize("tools", [None, [{"type": "function", "function": {"name": "get_weather"}}]])
def test_chat_stream_sends_reasoning_deltas(model, tools):
    model(*(TOOL if tools else ANSWER))
    events = sse(client.post("/v1/chat/completions", json={
        "messages": HI, "stream": True, "tools": tools}))
    deltas = [e["choices"][0]["delta"] for e in events if e.get("choices")]
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content") or "" for d in deltas)
    assert reasoning == ("Need weather." if tools else "Count the words.")
    assert "<tool_call>" not in content and "weather." not in content
    if tools:
        assert content == "Checking."
        assert [d for d in deltas if d.get("tool_calls")][0]["tool_calls"][0]["function"]["name"] == "get_weather"
    else:
        assert content == "Hello there."


def test_anthropic_thinking_block_comes_first_with_a_signature(model):
    model(*TOOL)
    body = client.post("/v1/messages", json={"messages": HI, "max_tokens": 50}).json()
    kinds = [b["type"] for b in body["content"]]
    assert kinds == ["thinking", "text", "tool_use"]
    thinking = body["content"][0]
    assert thinking["thinking"] == "Need weather."
    assert thinking["signature"] == anth.signature("Need weather.")
    assert body["content"][1]["text"] == "Checking."


def test_anthropic_stream_has_thinking_text_and_tool_blocks_in_order(model):
    model(*TOOL)
    events = sse(client.post("/v1/messages", json={"messages": HI, "max_tokens": 50, "stream": True}))
    starts = [(e["index"], e["content_block"]["type"]) for e in events
              if e["type"] == "content_block_start"]
    assert starts == [(0, "thinking"), (1, "text"), (2, "tool_use")]
    stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
    assert stops == [0, 1, 2]
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    assert "".join(d.get("thinking", "") for d in deltas) == "Need weather."
    assert "".join(d.get("text", "") for d in deltas).strip() == "Checking."
    signatures = [d["signature"] for d in deltas if d["type"] == "signature_delta"]
    assert signatures == [anth.signature("Need weather.")]
    # the signature closes the thinking block, before the text block opens
    order = [(e["type"], e.get("index"), e.get("delta", {}).get("type")) for e in events]
    assert order.index(("content_block_delta", 0, "signature_delta")) < order.index(
        ("content_block_stop", 0, None))


def test_anthropic_stream_without_reasoning_starts_with_text(model):
    model("</think>Bonjour.")
    events = sse(client.post("/v1/messages", json={"messages": HI, "max_tokens": 50, "stream": True}))
    starts = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
    assert starts == ["text"]


def test_anthropic_thinking_sent_back_is_ignored():
    body = anth.to_openai_body({"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hm", "signature": "x"},
            {"type": "text", "text": "Hello."}]}]})
    assert body["messages"][1]["content"] == [{"type": "text", "text": "Hello."}]


def test_responses_output_has_a_reasoning_item_first(model):
    model(*TOOL)
    output = client.post("/v1/responses", json={"input": "hi"}).json()["output"]
    assert [o["type"] for o in output] == ["reasoning", "message", "function_call"]
    assert output[0]["content"] == [{"type": "reasoning_text", "text": "Need weather."}]
    assert output[1]["content"][0]["text"] == "Checking."


def test_responses_stream_reasoning_events_and_indexes(model):
    model(*TOOL)
    events = sse(client.post("/v1/responses", json={"input": "hi", "stream": True}))
    added = [(e["output_index"], e["item"]["type"]) for e in events
             if e["type"] == "response.output_item.added"]
    assert added == [(0, "reasoning"), (1, "message")]
    reasoning = "".join(e["delta"] for e in events if e["type"] == "response.reasoning_text.delta")
    text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    assert reasoning == "Need weather." and text.strip() == "Checking."
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert [d["type"] for d in done] == ["reasoning", "message", "function_call"]
    ids = {e["item"]["type"]: e["item"]["id"] for e in events
           if e["type"] == "response.output_item.added"}
    assert done[0]["id"] == ids["reasoning"] and done[1]["id"] == ids["message"]
    final = [e for e in events if e["type"] == "response.completed"][0]["response"]
    assert [o["type"] for o in final["output"]] == ["reasoning", "message", "function_call"]


def test_responses_stream_flushes_a_short_answer(model):
    model("</think>Paris")
    events = sse(client.post("/v1/responses", json={"input": "hi", "stream": True}))
    text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    assert text == "Paris"
    assert not [e for e in events if e["type"] == "response.output_item.added"
                and e["item"]["type"] == "reasoning"]


INKLING = ("<|message_model|><|content_thinking|>Need use ", "get_weather tool.<|end_message|>"
           "<|message_model|>get_weather<|content_invoke_tool_json|>",
           '{"name":"get_weather","args":{"city":"Lomé","days":3}}<|end_message|>')


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/messages", "/v1/responses"])
def test_inkling_reasoning_and_tool_call_on_every_dialect(monkeypatch, route):
    monkeypatch.setattr(serve_app, "submit",
                        lambda *_a: generated(*INKLING, parser=InklingParser()))
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    body = {"messages": HI, "input": "hi", "max_tokens": 50, "tools": tools}
    out = client.post(route, json=body).json()
    if route == "/v1/chat/completions":
        message = out["choices"][0]["message"]
        assert message["reasoning_content"] == "Need use get_weather tool."
        assert message["content"] is None
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "Lomé", "days": 3}
    elif route == "/v1/messages":
        assert [b["type"] for b in out["content"]] == ["thinking", "tool_use"]
        assert out["content"][1]["input"] == {"city": "Lomé", "days": 3}
    else:
        assert [o["type"] for o in out["output"]] == ["reasoning", "function_call"]


UNPARSED = ("</think>", '<tool_call>\n{{"name": "get_weather"}}')


def test_streams_send_an_unparsed_call_as_text_like_the_plain_response(model):
    model(*UNPARSED)
    plain = client.post("/v1/messages", json={"messages": HI, "max_tokens": 50}).json()
    events = sse(client.post("/v1/messages", json={"messages": HI, "max_tokens": 50, "stream": True}))
    text = "".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta")
    assert text.strip() == plain["content"][0]["text"]

    events = sse(client.post("/v1/responses", json={"input": "hi", "stream": True}))
    text = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
    final = [e for e in events if e["type"] == "response.completed"][0]["response"]["output"]
    assert text.strip() == final[0]["content"][0]["text"] == plain["content"][0]["text"]
