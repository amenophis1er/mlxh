import queue
from types import SimpleNamespace as NS

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
