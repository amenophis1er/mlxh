import queue
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from mlxh import serve_app


client = TestClient(serve_app.app)


def queued(*items):
    result = queue.Queue()
    for item in (*items, None):
        result.put(item)
    return result


def generation(text="A", token=10, generation_tokens=1, **extra):
    return NS(
        text=text,
        token=token,
        prompt_tokens=5,
        generation_tokens=generation_tokens,
        finish_reason="stop",
        **extra,
    )


def test_shape_logprobs_string_and_utf8_modes():
    decoded = {10: "A", 11: "é", 12: "C"}
    entry = serve_app.shape_logprobs(
        10, -0.1, [10, 11, 12], [-0.1, -1.2, -2.3], decoded.__getitem__, False
    )

    assert entry["token"] == "A"
    assert entry["bytes"] == [65]
    assert [item["token"] for item in entry["top_logprobs"]] == ["A", "é", "C"]
    assert entry["top_logprobs"][1]["bytes"] == [195, 169]


def test_shape_logprobs_token_id_mode():
    entry = serve_app.shape_logprobs(
        10, -0.1, [11, 10], [-0.2, -0.3], lambda _token_id: "unused", True
    )

    assert entry["token"] == "token_id:10"
    assert entry["bytes"] is None
    assert [item["token"] for item in entry["top_logprobs"]] == [
        "token_id:11", "token_id:10",
    ]
    assert all(item["bytes"] is None for item in entry["top_logprobs"])


def test_stop_record_never_shapes_eos_even_when_it_flushes_text():
    response = generation(text="buffered", generation_tokens=2)
    response.finish_reason = "stop"

    assert not serve_app._should_shape_logprobs(response, previous_tokens=1)


@pytest.mark.parametrize("fields, field", [
    ({"top_logprobs": 129}, "top_logprobs"),
    ({"top_logprobs": -1}, "top_logprobs"),
    ({"top_logprobs": True}, "top_logprobs"),
    ({"logprob_token_ids": [True]}, "logprob_token_ids"),
    ({"logprob_token_ids": [1, 1]}, "logprob_token_ids"),
    ({"logprob_token_ids": []}, "logprob_token_ids"),
    ({"allowed_token_ids": [-1]}, "allowed_token_ids"),
    ({"allowed_token_ids": [2, 2]}, "allowed_token_ids"),
    ({"return_tokens_as_token_ids": 1}, "return_tokens_as_token_ids"),
    ({"stream": True}, "logprobs"),
])
def test_logprobs_validation_rejects_bad_requests(monkeypatch, fields, field):
    monkeypatch.setattr(
        serve_app, "submit",
        lambda _body: pytest.fail("invalid request reached submit"),
    )
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": True,
        **fields,
    })

    assert response.status_code == 400
    assert field in response.json()["detail"]


def test_chat_completion_has_stable_null_logprobs(monkeypatch):
    monkeypatch.setattr(serve_app, "submit", lambda _body: queued(generation()))

    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
    })

    assert response.status_code == 200
    assert response.json()["choices"][0]["logprobs"] is None


def test_chat_completion_assembles_one_entry_per_generated_token(monkeypatch):
    first = generation(
        logprobs_out=([10, 12, 11], [-0.1, -0.5, -1.0], -0.1),
    )
    final_duplicate = generation(text="", generation_tokens=1)
    monkeypatch.setattr(
        serve_app, "runner",
        NS(decode_token=lambda token_id: {10: "A", 11: "B", 12: "C"}[token_id]),
    )
    monkeypatch.setattr(
        serve_app, "submit", lambda _body: queued(first, final_duplicate)
    )

    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "choose"}],
        "logprobs": True,
        "top_logprobs": 2,
        "logprob_token_ids": [11],
    })

    assert response.status_code == 200
    content = response.json()["choices"][0]["logprobs"]["content"]
    assert len(content) == 1
    assert [item["token"] for item in content[0]["top_logprobs"]] == [
        "A", "C", "B",
    ]
    assert [item["logprob"] for item in content[0]["top_logprobs"]] == [
        -0.1, -0.5, -1.0,
    ]


def test_logprob_extensions_are_ignored_when_logprobs_is_false(monkeypatch):
    captured = {}

    def submit(body):
        captured.update(body)
        return queued(generation())

    monkeypatch.setattr(serve_app, "submit", submit)
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": False,
        "top_logprobs": 999,
        "logprob_token_ids": [],
        "_logprobs": {"untrusted": True},
    })

    assert response.status_code == 200
    assert "_logprobs" not in captured
    assert response.json()["choices"][0]["logprobs"] is None


def test_enable_thinking_request_override(monkeypatch):
    seen = {}

    class Runner:
        def template(self, messages, num_images, tools, thinking):
            seen["thinking"] = thinking
            return "prompt"

        def stream(self, *_args, **_kwargs):
            yield generation()

    monkeypatch.setattr(serve_app, "runner", Runner())
    monkeypatch.setitem(serve_app.SETTINGS, "thinking", "on")

    list(serve_app.run_generation({
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }))

    assert seen["thinking"] is False
