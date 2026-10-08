import queue
import pytest
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from mlxh import serve_app
from mlxh.engine import (
    EngineError,
    GenerationRequest,
    GenerationTerminal,
    InferenceEngine,
    Job,
    ReasoningParser,
    over_prompt_limit_message,
)


SETTINGS = {
    "max_queued": 4, "max_tokens_cap": 100, "memory_limit_gb": -1,
    "cache_limit_gb": 0, "gen_timeout_s": 0, "max_prompt_tokens": 0,
    "prompt_cache": False, "thinking": "auto",
}


def test_reasoning_parser_handles_preopened_and_split_inline_tags():
    parser = ReasoningParser(preopened=True)
    assert parser.feed("secret</thi") == ("", "secret")
    assert parser.feed("nk>\nanswer") == ("answer", "")

    parser = ReasoningParser()
    answer, reasoning = [], []
    for chunk in ("before<th", "ink>why", "</think>after"):
        visible, thought = parser.feed(chunk)
        answer.append(visible)
        reasoning.append(thought)
    visible, thought = parser.feed("", final=True)
    assert "".join(answer) + visible == "beforeafter"
    assert "".join(reasoning) + thought == "why"

    parser = ReasoningParser()
    assert parser.feed("before<think>why</think>\nafter", final=True) == (
        "beforeafter", "why"
    )
    assert parser.events == [
        ("text", "before"), ("reasoning", "why"), ("text", "after")
    ]


def test_engine_snapshot_counts_every_source_on_one_queue():
    engine = InferenceEngine("/unused", "model", dict(SETTINGS))
    engine.ready.set()
    first = engine.submit(GenerationRequest([], "chat"))
    second = engine.submit(GenerationRequest([], "openai"))

    snapshot = engine.snapshot()["runtime"]
    assert snapshot["requests"] == 2
    assert snapshot["queue_depth"] == 2
    assert engine.cancel(first.request_id)
    assert not engine.cancel(second.request_id)  # public jobs cannot be cancelled here


def test_private_generate_streams_reasoning_text_tools_and_usage(monkeypatch):
    class FakeEngine:
        model_id = "bonsai2"

        def submit(self, request):
            assert request.source == "chat"
            job = Job(request, queue.Queue())
            job.out.put(NS(text="answer", reasoning_text="thought"))
            job.out.put(NS(text="<tool_call><function=clock></function></tool_call>",
                           reasoning_text=""))
            job.out.put(GenerationTerminal("tool_calls", "completed", 7, 3, 0.5, 6.0))
            job.out.put(None)
            return job

        def cancel(self, _request_id):
            return True

    monkeypatch.setattr(serve_app, "engine", FakeEngine())
    response = TestClient(serve_app.app).post(
        "/mlxh/generate", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 200
    assert "event: reasoning_delta" in response.text
    assert '"text": "answer"' in response.text
    assert "event: tool_calls" in response.text
    assert '"output_tokens": 3' in response.text


def test_private_generate_rejects_declared_oversized_body_before_submit(monkeypatch):
    monkeypatch.setattr(
        serve_app, "engine",
        NS(submit=lambda _request: (_ for _ in ()).throw(AssertionError("submitted"))),
    )
    response = TestClient(serve_app.app).post(
        "/mlxh/generate", content=b"{}",
        headers={"content-type": "application/json",
                 "content-length": str(serve_app.PRIVATE_BODY_LIMIT + 1)},
    )
    assert response.status_code == 413


def test_public_timeout_is_not_reported_as_normal_completion(monkeypatch):
    def timed_out(*_args):
        result = queue.Queue()
        result.put(NS(text="partial", prompt_tokens=8, generation_tokens=4,
                      finish_reason=None))
        result.put(GenerationTerminal("timeout", "timed_out", 8, 4, 2.0, 2.0))
        result.put(None)
        return result

    monkeypatch.setattr(serve_app, "submit", timed_out)
    client = TestClient(serve_app.app)
    chat = client.post("/v1/chat/completions", json={"messages": []}).json()
    assert chat["choices"][0]["finish_reason"] == "length"
    assert chat["usage"]["completion_tokens"] == 4

    response = client.post("/v1/responses", json={"input": "hi"}).json()
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}


def test_resolve_context_window_detects_overrides_and_disables():
    engine = InferenceEngine("/unused", "test", {**SETTINGS})
    engine.runner = NS(context_window=32768)
    assert engine.context_window is None  # resolution happens on load
    assert engine._resolve_context_window() == 32768  # 0 = detect
    engine.settings["context_window"] = -1
    assert engine._resolve_context_window() is None  # -1 = off
    engine.settings["context_window"] = 8192
    assert engine._resolve_context_window() == 8192  # explicit override wins


def test_resolve_context_window_without_setting_or_detection():
    # settings without the key (older fixtures) behave like 0 = detect
    engine = InferenceEngine("/unused", "test", {k: v for k, v in SETTINGS.items()})
    engine.runner = NS(context_window=None)
    assert engine._resolve_context_window() is None
    engine.runner = NS()
    assert engine._resolve_context_window() is None


def test_snapshot_reports_context_window():
    engine = InferenceEngine("/unused", "test", {**SETTINGS})
    engine.context_window = 262144
    assert engine.snapshot()["context_window"] == 262144
    engine.context_window = None
    assert engine.snapshot()["context_window"] is None


def test_models_endpoint_reports_detected_context_length(monkeypatch):
    monkeypatch.setattr(serve_app, "engine", NS(context_window=32768))
    data = TestClient(serve_app.app).get("/v1/models").json()["data"]
    assert data[0]["context_length"] == 32768


def test_models_endpoint_omits_unknown_context_length(monkeypatch):
    monkeypatch.setattr(serve_app, "engine", NS(context_window=None))
    data = TestClient(serve_app.app).get("/v1/models").json()["data"]
    assert "context_length" not in data[0]


class EchoRunner:
    """Template yields 5 prompt tokens; streams one short completion."""
    supports_images = False
    supports_cache = False

    def template(self, messages, num_images=0, tools=None, thinking=None):
        return [1, 2, 3, 4, 5]

    def stream(self, prompt, **_kwargs):
        yield NS(text="ok", finish_reason="stop", generation_tokens=2,
                 generation_tps=1.0, prompt_tokens=5)


def _engine(**overrides):
    settings = {**SETTINGS, **overrides}
    engine = InferenceEngine("/unused", "test", settings)
    engine.runner = EchoRunner()
    return engine


def _request():
    return GenerationRequest(messages=[{"role": "user", "content": "hi"}],
                             source="chat")


def test_prompt_over_context_window_is_rejected(monkeypatch=None):
    engine = _engine(max_prompt_tokens=0)
    engine.context_window = 4
    with pytest.raises(EngineError) as raised:
        list(engine._run_generation(_request()))
    assert raised.value.status_code == 400
    assert "context window=4" in raised.value.detail
    assert "context_window" in raised.value.detail


def test_prompt_at_context_window_is_accepted():
    engine = _engine(max_prompt_tokens=0)
    engine.context_window = 5
    responses = list(engine._run_generation(_request()))
    assert responses[-1].text == "ok"


def test_max_prompt_tokens_precedes_context_window():
    engine = _engine(max_prompt_tokens=3)
    engine.context_window = 32768
    with pytest.raises(EngineError) as raised:
        list(engine._run_generation(_request()))
    assert "max_prompt_tokens=3" in raised.value.detail
    assert "context window" not in raised.value.detail


def test_window_tighter_than_max_prompt_tokens_precedes_it():
    engine = _engine(max_prompt_tokens=8192)
    engine.context_window = 4
    with pytest.raises(EngineError) as raised:
        list(engine._run_generation(_request()))
    assert "context window=4" in raised.value.detail
    assert "max_prompt_tokens" not in raised.value.detail


def test_no_limits_generates():
    engine = _engine(max_prompt_tokens=0)
    assert list(engine._run_generation(_request()))[-1].text == "ok"


def test_over_prompt_limit_message_chooses_between_limits():
    assert "max_prompt_tokens=8192" in over_prompt_limit_message(9000, 8192, None)
    assert "max_prompt_tokens=8192" in over_prompt_limit_message(9000, 8192, 32768)
    assert "max_prompt_tokens=32768" in over_prompt_limit_message(40000, 32768, 32768)
    assert "max_prompt_tokens=16384" in over_prompt_limit_message(40000, 16384, 32768)
    assert "context window=32768" in over_prompt_limit_message(40000, 65536, 32768)
    assert "context window=32768" in over_prompt_limit_message(40000, 0, 32768)


def test_over_window_prompt_surfaces_as_http_400(monkeypatch):
    result = queue.Queue()
    result.put(EngineError(400, "prompt is ~9 tokens, over the model "
                              "context window=8."))
    result.put(None)
    monkeypatch.setattr(serve_app, "submit", lambda _body, source="openai": result)
    response = TestClient(serve_app.app).post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert "context window=8" in response.json()["detail"]
