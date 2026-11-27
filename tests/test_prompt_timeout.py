"""A prompt still being processed when gen_timeout_s passes (no MLX needed)."""
import queue
import sys
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from mlxh import loader, serve_app
from mlxh.engine import EngineError, GenerationRequest, GenerationTerminal, InferenceEngine
from mlxh.loader import PromptTimeout, VLMRunner

HI = [{"role": "user", "content": "hi"}]


class LanguageModel:
    """What mlx-vlm calls once per chunk of the prompt, then once per token."""
    calls = 0

    def __call__(self, inputs=None, cache=None):
        self.calls += 1
        return "logits"


def vlm(monkeypatch, chunks=3, tokens=2):
    runner = VLMRunner(NS(language_model=LanguageModel()), None, None)

    def stream_generate(model, processor, prompt, **_kw):
        for _ in range(chunks):
            model.language_model(inputs=prompt)
        for i in range(tokens):
            model.language_model(inputs=prompt)
            yield NS(text=str(i))

    monkeypatch.setitem(sys.modules, "mlx_vlm", NS(stream_generate=stream_generate))
    return runner


def test_watching_keeps_the_models_class_identity():
    runner = VLMRunner(NS(language_model=LanguageModel()), None, None)
    watched = type(runner.model.language_model)
    assert watched is not LanguageModel and issubclass(watched, LanguageModel)
    assert (watched.__name__, watched.__module__) == (LanguageModel.__name__, LanguageModel.__module__)
    assert runner.model.language_model(inputs=[1]) == "logits"


def test_vlm_prompt_past_its_deadline_stops_at_the_next_chunk(monkeypatch):
    runner = vlm(monkeypatch)
    late_after = iter([False, True])
    monkeypatch.setattr(loader, "_past", lambda deadline: deadline is not None and next(late_after))
    with pytest.raises(PromptTimeout):
        list(runner.stream("prompt", deadline=5.0))
    assert runner.model.language_model.calls == 1, "one chunk went through, the next was refused"
    assert runner._deadline is None


def test_vlm_deadline_guards_the_prompt_not_the_answer(monkeypatch):
    runner = vlm(monkeypatch, chunks=2, tokens=3)
    seen = []

    def past(deadline):
        seen.append(deadline)
        return False

    monkeypatch.setattr(loader, "_past", past)
    stream = runner.stream("prompt", deadline=5.0)
    assert next(stream).text == "0"
    monkeypatch.setattr(loader, "_past", lambda deadline: deadline is not None)  # now it is late
    assert [r.text for r in stream] == ["1", "2"], "a token is out: the engine decides from here"
    assert seen == [5.0, 5.0, 5.0]  # two prompt chunks and the first token's pass


def test_vlm_without_a_deadline_or_a_language_model_is_untouched(monkeypatch):
    runner = vlm(monkeypatch)
    monkeypatch.setattr(loader, "_past", lambda deadline: deadline is not None)
    assert len(list(runner.stream("prompt"))) == 2
    assert VLMRunner(NS(), None, None)._deadline is None


def test_the_engine_gives_the_runner_the_requests_deadline():
    seen = {}

    class Runner:
        def template(self, messages, **_kw):
            return "prompt"

        def stream(self, *_a, **kw):
            seen.update(kw)
            return iter(())

    engine = InferenceEngine("/unused", "m", {**serve_app.SETTINGS, "max_prompt_tokens": -1})
    engine.runner = Runner()
    list(engine._run_generation(GenerationRequest(HI, "openai"), deadline=42.0))
    assert seen["deadline"] == 42.0
    list(engine._run_generation(GenerationRequest(HI, "openai")))
    assert seen["deadline"] is None


def test_the_error_says_how_far_it_got_and_what_to_change():
    error = InferenceEngine._prompt_timeout_error(PromptTimeout(126976, 250000), 600)
    assert error.status_code == 504
    assert "gen_timeout_s=600s passed after 126,976 of 250,000 tokens" in error.detail
    assert "mlxh config gen_timeout_s" in error.detail
    assert "after" not in InferenceEngine._prompt_timeout_error(PromptTimeout(), 600).detail.split("passed")[1][:8]


def test_engine_turns_a_prompt_timeout_into_a_504_and_a_timed_out_request(monkeypatch):
    pytest.importorskip("mlx.core", reason="the engine thread configures MLX")
    from mlxh import engine as engine_mod

    class Runner:
        supports_images = False

        def template(self, messages, **_kw):
            return [1, 2, 3]

        def stream(self, prompt, deadline=None, **_kw):
            assert deadline is not None
            raise PromptTimeout(2048, 9000)
            yield

    monkeypatch.setattr(engine_mod, "load_runner", lambda _path: Runner())
    engine = InferenceEngine("/unused", "m", {**serve_app.SETTINGS, "gen_timeout_s": 600,
                                              "max_prompt_tokens": -1, "memory_limit_gb": -1,
                                              "prompt_cache": False})
    engine.start()
    out = engine.submit(GenerationRequest(HI, "openai")).out
    items = list(iter(lambda: out.get(timeout=30), None))
    error = [i for i in items if isinstance(i, EngineError)][0]
    terminal = [i for i in items if isinstance(i, GenerationTerminal)][0]
    assert error.status_code == 504 and "2,048 of 9,000 tokens" in error.detail
    assert (terminal.outcome, terminal.finish_reason) == ("timed_out", "timeout")
    assert engine.snapshot()["runtime"]["last_request"]["outcome"] == "timed_out"


@pytest.mark.parametrize("route,body", [
    ("/v1/chat/completions", {"messages": HI}),
    ("/v1/messages", {"messages": HI, "max_tokens": 9}),
    ("/v1/responses", {"input": "hi"}),
])
def test_every_dialect_answers_504_with_the_reason(monkeypatch, route, body):
    def timed_out(*_a):
        out = queue.Queue()
        out.put(InferenceEngine._prompt_timeout_error(PromptTimeout(10, 20), 600))
        out.put(GenerationTerminal("timeout", "timed_out", 0, 0, 600.0, 0.0))
        out.put(None)
        return out

    monkeypatch.setattr(serve_app, "submit", timed_out)
    response = TestClient(serve_app.app).post(route, json=body)
    assert response.status_code == 504
    assert "still being processed" in response.json()["detail"]
