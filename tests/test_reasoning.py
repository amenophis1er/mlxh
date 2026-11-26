from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from mlxh import anthropic_compat as anth
from mlxh import loader, reasoning, serve_app
from mlxh import responses_compat as oresp
from mlxh.engine import InferenceEngine

client = TestClient(serve_app.app)
HI = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("value", reasoning.LEVELS)
def test_every_openai_level_is_accepted(value):
    request = serve_app._request({"messages": HI, "reasoning_effort": value})
    assert request.reasoning_effort == value
    assert request.thinking is (value != "none")


def test_unknown_effort_is_rejected():
    with pytest.raises(reasoning.ReasoningError, match="reasoning_effort must be one of"):
        serve_app._request({"messages": HI, "reasoning_effort": "extreme"})


def test_no_effort_leaves_enable_thinking_in_charge():
    request = serve_app._request({
        "messages": HI, "chat_template_kwargs": {"enable_thinking": False}})
    assert request.reasoning_effort is None and request.thinking is False


def test_effort_wins_over_enable_thinking():
    request = serve_app._request({
        "messages": HI, "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": False}})
    assert request.thinking is True and request.reasoning_effort == "low"


@pytest.mark.parametrize("route,body", [
    ("/v1/chat/completions", {"messages": HI, "reasoning_effort": "turbo"}),
    ("/v1/responses", {"input": "hi", "reasoning": {"effort": "turbo"}}),
    ("/v1/responses", {"input": "hi", "reasoning": "high"}),
    ("/v1/messages", {"messages": HI, "max_tokens": 9, "thinking": {"type": "on"}}),
    ("/v1/messages", {"messages": HI, "max_tokens": 9, "thinking": {"type": "enabled"}}),
])
def test_bad_reasoning_values_are_a_400(monkeypatch, route, body):
    monkeypatch.setattr(serve_app, "engine", NS(submit=lambda _r: pytest.fail("submitted")))
    response = client.post(route, json=body)
    assert response.status_code == 400
    assert "must" in response.json()["detail"]


def test_responses_reasoning_effort_reaches_the_body():
    body = oresp.to_openai_body({"input": "hi", "reasoning": {"effort": "minimal"}})
    assert body["reasoning_effort"] == "minimal"
    assert "reasoning_effort" not in oresp.to_openai_body(
        {"input": "hi", "reasoning": {"summary": "auto"}})


@pytest.mark.parametrize("thinking,effort", [
    ({"type": "disabled"}, "none"),
    ({"type": "enabled", "budget_tokens": 1024}, "low"),
    ({"type": "enabled", "budget_tokens": 4095}, "low"),
    ({"type": "enabled", "budget_tokens": 4096}, "medium"),
    ({"type": "enabled", "budget_tokens": 16383}, "medium"),
    ({"type": "enabled", "budget_tokens": 16384}, "high"),
    ({"type": "enabled", "budget_tokens": 64000}, "high"),
])
def test_anthropic_thinking_maps_to_a_level(thinking, effort):
    body = anth.to_openai_body({"messages": HI, "thinking": thinking})
    assert body["reasoning_effort"] == effort


@pytest.mark.parametrize("thinking", [None, {"type": "adaptive"}])
def test_anthropic_default_thinking_sends_no_level(thinking):
    body = anth.to_openai_body({"messages": HI, "thinking": thinking})
    assert "reasoning_effort" not in body


@pytest.mark.parametrize("budget", [0, -5, "4096", True, 2.5])
def test_anthropic_bad_budget_is_rejected(budget):
    with pytest.raises(reasoning.ReasoningError, match="budget_tokens"):
        reasoning.from_anthropic({"type": "enabled", "budget_tokens": budget})


class Tok:
    def __init__(self):
        self.seen = {}

    def apply_chat_template(self, messages, **kw):
        self.seen = kw
        return [1]


@pytest.mark.parametrize("thinking,effort,sent", [
    (None, None, None),
    (True, None, None),
    (False, None, "none"),
    (True, "low", "low"),
    (True, "xhigh", "xhigh"),
    (False, "none", "none"),
])
def test_inkling_passes_the_level_to_its_template(thinking, effort, sent):
    tok = Tok()
    loader.InklingRunner(None, tok).template(HI, thinking=thinking, effort=effort)
    assert tok.seen.get("reasoning_effort") == sent
    assert "enable_thinking" not in tok.seen


@pytest.mark.parametrize("effort,enabled", [("none", False), ("minimal", True), ("max", True)])
def test_other_runners_only_switch_thinking(effort, enabled):
    tok = Tok()
    request = serve_app._request({"messages": HI, "reasoning_effort": effort})
    loader.TextRunner(None, tok).template(HI, thinking=request.thinking,
                                          effort=request.reasoning_effort)
    assert tok.seen["enable_thinking"] is enabled
    assert "reasoning_effort" not in tok.seen


@pytest.mark.parametrize("setting,body,expected", [
    ("off", {"reasoning_effort": "high"}, (True, "high")),
    ("on", {"reasoning_effort": "none"}, (False, "none")),
    ("off", {}, (False, None)),
    ("auto", {"thinking_kw": True}, (True, None)),
])
def test_request_effort_wins_over_the_setting(setting, body, expected):
    seen = {}

    class Runner:
        def template(self, messages, num_images, tools, thinking, effort=None):
            seen["args"] = (thinking, effort)
            return "prompt"

        def stream(self, *_args, **_kwargs):
            return iter(())

    if body.pop("thinking_kw", None):
        body["chat_template_kwargs"] = {"enable_thinking": True}
    engine = InferenceEngine("/unused", "test", {**serve_app.SETTINGS, "thinking": setting})
    engine.runner = Runner()
    list(engine._run_generation(serve_app._request({"messages": HI, **body})))
    assert seen["args"] == expected
