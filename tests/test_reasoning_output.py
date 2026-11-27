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


# --- stream endings, event fields, empty reasoning --------------------------

def anthropic_stream(**body):
    return sse(client.post("/v1/messages", json={"messages": HI, "max_tokens": 50,
                                                 "stream": True, **body}))


def responses_stream():
    return sse(client.post("/v1/responses", json={"input": "hi", "stream": True}))


def texts(events, kind):
    return "".join(e["delta"] for e in events if e["type"] == kind)


@pytest.mark.parametrize("chunks,plain_text", [
    (("</think>", CALL, "\nI have asked for the weather."), "I have asked for the weather."),
    (("</think>Before. ", CALL, " After the call."), "Before.  After the call."),
    (("</think> \n", CALL), None),
])
def test_text_around_a_call_is_streamed_as_the_plain_response_has_it(model, chunks, plain_text):
    model(*chunks)
    plain = client.post("/v1/responses", json={"input": "hi"}).json()["output"]
    assert [o["content"][0]["text"] for o in plain if o["type"] == "message"] == (
        [plain_text] if plain_text else [])

    events = responses_stream()
    final = [e for e in events if e["type"] == "response.completed"][0]["response"]["output"]
    assert [(o["type"], o.get("content")) for o in final] == [
        (o["type"], o.get("content")) for o in plain]
    assert texts(events, "response.output_text.delta").split() == (plain_text or "").split()
    added = [e["item"]["type"] for e in events if e["type"] == "response.output_item.added"]
    assert added == (["message"] if plain_text else [])

    events = anthropic_stream()
    text = "".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta")
    assert text.split() == (plain_text or "").split()
    starts = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
    assert starts == (["text", "tool_use"] if plain_text else ["tool_use"])


@pytest.mark.parametrize("split", range(1, len("<tool_call>")))
def test_a_marker_split_across_chunks_never_reaches_the_stream(model, split):
    marker, rest = CALL[:split], CALL[split:]
    model("</think>Checking. " + marker, rest)
    assert texts(responses_stream(), "response.output_text.delta") == "Checking. "
    events = anthropic_stream()
    assert "".join(e["delta"].get("text", "") for e in events
                   if e["type"] == "content_block_delta") == "Checking. "


def test_text_that_only_looks_like_a_marker_is_delivered(model):
    model("</think>Use the <tool", "_box> now.")
    assert texts(responses_stream(), "response.output_text.delta") == "Use the <tool_box> now."


@pytest.mark.parametrize("raw", [("\n\n</think>\n\nHello.",), ("  ", "\n</think>Hello.")])
def test_an_empty_think_block_is_not_reasoning(model, raw):
    model(*raw)
    message = client.post("/v1/chat/completions", json={"messages": HI}).json()["choices"][0]["message"]
    assert "reasoning_content" not in message and message["content"] == "Hello."
    deltas = [e["choices"][0]["delta"] for e in sse(client.post(
        "/v1/chat/completions", json={"messages": HI, "stream": True})) if e.get("choices")]
    assert not any("reasoning_content" in d for d in deltas)

    plain = client.post("/v1/messages", json={"messages": HI, "max_tokens": 50}).json()
    assert [b["type"] for b in plain["content"]] == ["text"]
    starts = [e["content_block"]["type"] for e in anthropic_stream()
              if e["type"] == "content_block_start"]
    assert starts == ["text"]

    assert [o["type"] for o in client.post("/v1/responses", json={"input": "hi"}).json()["output"]] == ["message"]
    added = [e["item"]["type"] for e in responses_stream() if e["type"] == "response.output_item.added"]
    assert added == ["message"]


def test_reasoning_that_starts_with_blank_lines_is_kept_whole(model):
    model("\n", "\nWhy.</think>Hi.")
    events = responses_stream()
    assert texts(events, "response.reasoning_text.delta") == "\n\nWhy."
    message = client.post("/v1/chat/completions", json={"messages": HI}).json()["choices"][0]["message"]
    assert message["reasoning_content"] == "\n\nWhy."


def test_responses_stream_events_all_point_at_their_item(model):
    model(*TOOL)
    events = responses_stream()
    added = {e["item"]["type"]: (e["item"]["id"], e["output_index"])
             for e in events if e["type"] == "response.output_item.added"}
    assert added["reasoning"][1] == 0 and added["message"][1] == 1
    assert [e["item"]["status"] for e in events if e["type"] == "response.output_item.added"] == [
        "in_progress", "in_progress"]
    per_item = {
        "reasoning": ["response.content_part.added", "response.reasoning_text.delta",
                      "response.reasoning_text.done", "response.content_part.done"],
        "message": ["response.content_part.added", "response.output_text.delta",
                    "response.output_text.done", "response.content_part.done"],
    }
    for kind, wanted in per_item.items():
        mine = [e for e in events if e.get("item_id") == added[kind][0]]
        assert [e["type"] for e in mine if "delta" not in e["type"]] == [
            w for w in wanted if "delta" not in w]
        assert wanted[1] in [e["type"] for e in mine]
        assert {e["output_index"] for e in mine} == {added[kind][1]}
        assert {e["content_index"] for e in mine} == {0}
    done = {e["type"]: e for e in events}
    assert done["response.reasoning_text.done"]["text"] == "Need weather."
    assert done["response.output_text.done"]["text"] == "Checking."
    parts = [e["part"] for e in events if e["type"] == "response.content_part.done"]
    assert parts == [{"type": "reasoning_text", "text": "Need weather."},
                     {"type": "output_text", "text": "Checking.", "annotations": []}]
    finished = [e for e in events if e["type"] == "response.output_item.done"]
    assert [e["output_index"] for e in finished] == [0, 1, 2]
    assert finished[0]["item"] == {"id": added["reasoning"][0], "type": "reasoning",
                                   "status": "completed", "summary": [],
                                   "content": [{"type": "reasoning_text", "text": "Need weather."}]}
    assert finished[1]["item"]["status"] == "completed"


def test_anthropic_thinking_events_have_the_documented_shape(model):
    model(*ANSWER)
    events = anthropic_stream()
    start = [e for e in events if e["type"] == "content_block_start"][0]
    assert start["content_block"] == {"type": "thinking", "thinking": "", "signature": ""}
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta" and e["index"] == 0]
    assert {d["type"] for d in deltas} == {"thinking_delta", "signature_delta"}
    assert deltas[-1]["type"] == "signature_delta"


def test_phases_within_one_chunk_keep_their_order(monkeypatch):
    # text then reasoning in a single chunk: only phase_deltas knows the order
    monkeypatch.setattr(serve_app, "submit", lambda *_a: generated(
        "Before.<think>why</think>", "After.", parser=ReasoningParser()))
    starts = [e["content_block"]["type"] for e in anthropic_stream()
              if e["type"] == "content_block_start"]
    assert starts == ["text", "thinking", "text"]


def private_text(monkeypatch, *chunks):
    from mlxh.engine import Job

    class Engine:
        model_id = "m"

        def submit(self, request):
            job = Job(request, queue.Queue())
            for chunk in chunks:
                job.out.put(NS(text=chunk, reasoning_text=""))
            job.out.put(GenerationTerminal("stop", "completed", 1, 1, 0.1, 1.0))
            job.out.put(None)
            return job

        def cancel(self, _request_id):
            return True

    monkeypatch.setattr(serve_app, "engine", Engine())
    response = client.post("/mlxh/generate", json={"messages": HI})
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: {")]
    return "".join(e["text"] for e in events if set(e) == {"text"})


def test_chat_transport_loses_no_text_after_leading_whitespace(monkeypatch):
    assert private_text(monkeypatch, "  \n", "Hello world, this is", " it") == "  \nHello world, this is it"
    assert private_text(monkeypatch, "Before. ", CALL, " After.") == "Before.  After."
