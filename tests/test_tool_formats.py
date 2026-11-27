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
