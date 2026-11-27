import queue
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from mlxh import serve_app
from mlxh.engine import (
    GenerationRequest,
    GenerationTerminal,
    InferenceEngine,
    Job,
    ReasoningParser,
)


SETTINGS = {
    "max_queued": 4, "max_tokens_cap": 100, "memory_limit_gb": -1,
    "cache_limit_gb": 0, "gen_timeout_s": 0, "max_prompt_tokens": -1,
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


def test_usage_reports_prompt_tokens_the_cache_supplied(monkeypatch):
    def cached(*_args):
        result = queue.Queue()
        result.put(NS(text="hi", prompt_tokens=50, generation_tokens=2,
                      cached_tokens=30, finish_reason="stop"))
        result.put(GenerationTerminal("stop", "completed", 50, 2, 0.1, 20.0, 30))
        result.put(None)
        return result

    monkeypatch.setattr(serve_app, "submit", cached)
    client = TestClient(serve_app.app)
    chat = client.post("/v1/chat/completions", json={"messages": []}).json()
    assert chat["usage"]["prompt_tokens"] == 50
    assert chat["usage"]["prompt_tokens_details"] == {"cached_tokens": 30}

    response = client.post("/v1/responses", json={"input": "hi"}).json()
    assert response["usage"]["input_tokens"] == 50
    assert response["usage"]["input_tokens_details"] == {"cached_tokens": 30}

    message = client.post("/v1/messages", json={
        "model": "m", "max_tokens": 5,
        "messages": [{"role": "user", "content": "hi"}]}).json()
    assert message["usage"] == {"input_tokens": 20, "output_tokens": 2,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 30}

    stream = client.post("/v1/messages", json={
        "model": "m", "max_tokens": 5, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]}).text
    assert '"cache_read_input_tokens": 30' in stream

    stream = client.post("/v1/responses", json={"input": "hi", "stream": True}).text
    assert '"input_tokens_details": {"cached_tokens": 30}' in stream

    stream = client.post("/v1/chat/completions", json={"messages": [], "stream": True}).text
    assert '"prompt_tokens_details": {"cached_tokens": 30}' in stream


def test_engine_carries_cached_tokens_into_the_terminal(monkeypatch):
    pytest.importorskip("mlx.core", reason="the engine thread configures MLX")
    from mlxh import engine as engine_mod

    class Runner:
        supports_images = False

        def template(self, messages, **_kw):
            return [1, 2, 3]

        def stream(self, prompt, **_kw):
            yield NS(text="hi", token=7, prompt_tokens=50, generation_tokens=1,
                     generation_tps=9.0, cached_tokens=30, finish_reason="stop")

    monkeypatch.setattr(engine_mod, "load_runner", lambda _path: Runner())
    engine = InferenceEngine("/unused", "m", dict(SETTINGS), exit_on_load_failure=False)
    engine.start()
    out = engine.submit(GenerationRequest([{"role": "user", "content": "hi"}], "openai")).out
    items = list(iter(lambda: out.get(timeout=30), None))
    terminal = [i for i in items if isinstance(i, GenerationTerminal)][0]
    assert (terminal.prompt_tokens, terminal.cached_tokens) == (50, 30)


def test_prompt_cache_cap_follows_setting_then_memory_limit(monkeypatch):
    monkeypatch.setattr("mlxh.cli.total_ram_bytes", lambda: 64 * 10**9)
    explicit = InferenceEngine("/unused", "m", {**SETTINGS, "prompt_cache_gb": 3})
    assert explicit._prompt_cache_bytes() == 3 * 10**9
    limited = InferenceEngine("/unused", "m", {**SETTINGS, "memory_limit_gb": 40})
    assert limited._prompt_cache_bytes() == 10 * 10**9
    auto = InferenceEngine("/unused", "m", {**SETTINGS, "memory_limit_gb": 0})
    assert auto._prompt_cache_bytes() == int(64 * 10**9 * 0.8 * 0.25)
    disabled = InferenceEngine("/unused", "m", dict(SETTINGS))  # memory_limit_gb=-1
    assert disabled._prompt_cache_bytes() == int(64 * 10**9 * 0.2)


def test_every_runner_shares_the_enable_cache_contract():
    seen = []
    engine = InferenceEngine("/unused", "m", {**SETTINGS, "prompt_cache_gb": 1})
    engine.runner = NS(enable_cache=lambda max_bytes: seen.append(max_bytes) or "on")
    engine._enable_prompt_cache()
    assert seen == [10**9]


def test_snapshot_reports_prefix_cache_only_for_runners_that_have_one():
    engine = InferenceEngine("/unused", "m", dict(SETTINGS))
    assert engine.snapshot()["prompt_cache"] is None
    engine.runner = NS(cache_stats=lambda: {"entries": 2}, supports_images=False)
    assert engine.snapshot()["prompt_cache"] == {"entries": 2}
