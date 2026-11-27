import json
import queue
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from mlxh import loader, serve_app
from mlxh.engine import GenerationTerminal
from mlxh.toolcalls import GEMMA4, HERMES, QWEN_XML

client = TestClient(serve_app.app)
HI = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]
GEMMA = ("Checking ", "now.<|tool", '_call>call:get_weather{city:<|"|>Lomé<|"|>,', 'days:<|"|>3<|"|>}<tool_call|>')
QWEN25 = ("Checking now.\n<tool", '_call>\n{{"name": "get_weather", ', '"arguments": {"city": "Lomé", "days": "3"}}')


@pytest.fixture
def model(monkeypatch):
    def use(fmt, chunks):
        def submit(*_a):
            out = queue.Queue()
            for i, text in enumerate(chunks):
                out.put(NS(text=text, prompt_tokens=4, generation_tokens=i + 1,
                           finish_reason="stop" if i == len(chunks) - 1 else None))
            out.put(GenerationTerminal("stop", "completed", 4, len(chunks), 1.0, 1.0))
            out.put(None)
            return out
        monkeypatch.setattr(serve_app, "submit", submit)
        monkeypatch.setattr(serve_app, "engine", NS(runner=NS(tool_format=fmt)))
    return use


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: {")]


@pytest.mark.parametrize("fmt,chunks", [(GEMMA4, GEMMA), (HERMES, QWEN25)])
def test_every_dialect_returns_the_call_with_values_as_written(model, fmt, chunks):
    model(fmt, chunks)
    expected = {"city": "Lomé", "days": "3"}
    chat = client.post("/v1/chat/completions", json={"messages": HI, "tools": TOOLS}).json()
    message = chat["choices"][0]["message"]
    assert message["content"] == "Checking now."
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == expected
    assert chat["choices"][0]["finish_reason"] == "tool_calls"

    anth = client.post("/v1/messages", json={"messages": HI, "max_tokens": 9}).json()
    assert anth["content"][-1]["input"] == expected and anth["stop_reason"] == "tool_use"

    resp = client.post("/v1/responses", json={"input": "hi"}).json()
    call = [o for o in resp["output"] if o["type"] == "function_call"][0]
    assert json.loads(call["arguments"]) == expected


@pytest.mark.parametrize("fmt,chunks", [(GEMMA4, GEMMA), (HERMES, QWEN25)])
def test_streams_stop_text_at_the_formats_marker(model, fmt, chunks):
    model(fmt, chunks)
    anth = events(client.post("/v1/messages", json={"messages": HI, "max_tokens": 9, "stream": True}))
    text = "".join(e["delta"].get("text", "") for e in anth if e["type"] == "content_block_delta")
    assert text.strip() == "Checking now."
    assert [e["content_block"]["type"] for e in anth if e["type"] == "content_block_start"] == ["text", "tool_use"]

    resp = events(client.post("/v1/responses", json={"input": "hi", "stream": True}))
    text = "".join(e["delta"] for e in resp if e["type"] == "response.output_text.delta")
    assert text.strip() == "Checking now."

    chat = events(client.post("/v1/chat/completions", json={"messages": HI, "tools": TOOLS, "stream": True}))
    deltas = [e["choices"][0]["delta"] for e in chat if e.get("choices")]
    assert "".join(d.get("content") or "" for d in deltas) == "Checking now."
    assert sum(len(d.get("tool_calls") or []) for d in deltas) == 1


def test_no_engine_means_qwen_xml(monkeypatch):
    monkeypatch.setattr(serve_app, "engine", None)
    assert serve_app._tool_format() is QWEN_XML


def test_load_picks_the_format_from_the_model_dir(monkeypatch, tmp_path):
    (tmp_path / "chat_template.jinja").write_text("{{ '<|tool_call>call:' }}")
    monkeypatch.setattr(loader, "_load_runner", lambda _d: loader.TextRunner(None, None))
    assert loader.load_runner(tmp_path).tool_format is GEMMA4


def test_inkling_keeps_qwen_xml_whatever_its_template_says(monkeypatch, tmp_path):
    (tmp_path / "chat_template.jinja").write_text("<tool_call>{json}</tool_call>")
    monkeypatch.setattr(loader, "_load_runner", lambda _d: loader.InklingRunner(None, None))
    assert loader.load_runner(tmp_path).tool_format is QWEN_XML


# --- review follow-ups -----------------------------------------------------

def private_events(monkeypatch, fmt, chunks):
    from mlxh.engine import Job

    class Engine:
        model_id = "m"
        runner = NS(tool_format=fmt)

        def submit(self, request):
            job = Job(request, queue.Queue())
            for chunk in chunks:
                job.out.put(NS(text=chunk, reasoning_text=""))
            job.out.put(GenerationTerminal("tool_calls", "completed", 1, 1, 0.1, 1.0))
            job.out.put(None)
            return job

        def cancel(self, _request_id):
            return True

    monkeypatch.setattr(serve_app, "engine", Engine())
    return events(client.post("/mlxh/generate", json={"messages": HI}))


@pytest.mark.parametrize("fmt,chunks", [(GEMMA4, GEMMA), (HERMES, QWEN25)])
def test_chat_transport_uses_the_models_format(monkeypatch, fmt, chunks):
    got = private_events(monkeypatch, fmt, chunks)
    text = "".join(e["text"] for e in got if set(e) == {"text"})
    calls = [e["calls"] for e in got if "calls" in e][0]
    assert text.strip() == "Checking now."
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Lomé", "days": "3"}


@pytest.mark.parametrize("fmt,chunks", [(GEMMA4, GEMMA), (HERMES, QWEN25)])
def test_responses_stream_carries_the_parsed_call(model, fmt, chunks):
    model(fmt, chunks)
    got = events(client.post("/v1/responses", json={"input": "hi", "stream": True}))
    done = [e["item"] for e in got if e["type"] == "response.output_item.done"]
    assert [d["type"] for d in done] == ["message", "function_call"]
    assert json.loads(done[1]["arguments"]) == {"city": "Lomé", "days": "3"}
    final = [e for e in got if e["type"] == "response.completed"][0]["response"]["output"]
    assert [o["type"] for o in final] == ["message", "function_call"]


@pytest.mark.parametrize("split", range(1, len(GEMMA4.marker)))
def test_gemma_marker_split_across_chunks_never_reaches_any_stream(model, monkeypatch, split):
    call = '<|tool_call>call:f{}<tool_call|>'
    chunks = ("Checking. " + call[:split], call[split:])
    model(GEMMA4, chunks)
    anth = events(client.post("/v1/messages", json={"messages": HI, "max_tokens": 9, "stream": True}))
    assert "".join(e["delta"].get("text", "") for e in anth
                   if e["type"] == "content_block_delta") == "Checking. "
    resp = events(client.post("/v1/responses", json={"input": "hi", "stream": True}))
    assert "".join(e["delta"] for e in resp if e["type"] == "response.output_text.delta") == "Checking. "
    got = private_events(monkeypatch, GEMMA4, chunks)
    assert "".join(e["text"] for e in got if set(e) == {"text"}) == "Checking. "


def test_the_engine_hands_the_template_history_in_the_formats_shape():
    from mlxh.engine import GenerationRequest, InferenceEngine
    seen = {}

    class Runner:
        tool_format = GEMMA4

        def template(self, messages, **_kw):
            seen["messages"] = messages
            return "prompt"

        def stream(self, *_a, **_kw):
            return iter(())

    engine = InferenceEngine("/unused", "m", dict(serve_app.SETTINGS))
    engine.runner = Runner()
    history = [{"role": "user", "content": "weather?"},
               {"role": "assistant", "content": "Checking.", "tool_calls": [
                   {"id": "c1", "function": {"name": "get_weather", "arguments": "{}"}}]},
               {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
    list(engine._run_generation(GenerationRequest(history, "openai")))
    assert [m["content"] for m in seen["messages"]] == ["weather?", "", "sunny"]
    assert seen["messages"][1]["tool_calls"][0]["id"] == "c1"


def test_chat_cli_names_each_tool_result_for_templates_that_pair_them(monkeypatch):
    from mlxh import chat_cli
    calls = [{"id": "call_a", "function": {"name": "get_weather", "arguments": "{}"}},
             {"id": "call_b", "function": {"name": "get_time", "arguments": "{}"}}]
    rounds = iter([("", calls, {}), ("Done.", None, {})])
    monkeypatch.setattr(chat_cli, "generate_once", lambda *_a, **_kw: next(rounds))
    messages = []
    chat_cli.ask(None, messages, [], 10, tools=({"get_weather": lambda: {"t": 1},
                                                 "get_time": lambda: {"h": 2}}, []))
    assert [c["id"] for c in messages[0]["tool_calls"]] == ["call_a", "call_b"]
    assert [(m["tool_call_id"], m["name"]) for m in messages if m["role"] == "tool"] == [
        ("call_a", "get_weather"), ("call_b", "get_time")]
