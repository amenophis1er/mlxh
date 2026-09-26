import json

from mlxh.chat_transport import ChatTransport


class Response:
    def __init__(self, lines):
        self.lines = iter(line.encode() for line in lines)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.lines)

    def close(self):
        self.closed = True


def test_transport_parses_comments_and_multiline_sse(monkeypatch):
    response = Response([
        ": ping\n", "\n",
        "event: start\n", 'data: {"request_id":\n', 'data: "req_1"}\n', "\n",
        "event: text_delta\n", 'data: {"text":"hi"}\n', "\n",
        "event: done\n", 'data: {"outcome":"completed"}\n', "\n",
    ])
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)
    transport = ChatTransport("http://127.0.0.1:1060")

    assert list(transport.generate({"messages": []})) == [
        ("start", {"request_id": "req_1"}),
        ("text_delta", {"text": "hi"}),
        ("done", {"outcome": "completed"}),
    ]
    assert response.closed
    assert transport.active_request_id is None


def test_transport_includes_selected_model_for_manager_routing(monkeypatch):
    captured = {}
    monkeypatch.setattr("urllib.request.urlopen", lambda request, **_kwargs: (
        captured.update(json.loads(request.data)) or Response([])
    ))
    transport = ChatTransport("http://127.0.0.1:1060", model="bonsai2")
    assert list(transport.generate({"messages": []})) == []
    assert captured["model"] == "bonsai2"


def test_transport_requests_capabilities_for_selected_model(monkeypatch):
    captured = {}

    class InfoResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return b'{"capabilities":{"images":true}}'

    def open_url(url, **_kwargs):
        captured["url"] = url
        return InfoResponse()

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    transport = ChatTransport("http://127.0.0.1:1060", model="vision model")
    assert transport.info()["capabilities"]["images"] is True
    assert captured["url"].endswith("?model=vision+model")
