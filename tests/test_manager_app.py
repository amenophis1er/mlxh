import json
import asyncio
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mlxh import manager_app as manager


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(manager, "CFG", {
        "worker_idle_timeout_s": 300, "models_dir": "/models",
    })
    monkeypatch.setattr(manager, "MODELS", {"alpha": "/models/alpha", "beta": "/models/beta"})
    from pathlib import Path
    monkeypatch.setattr("mlxh.cli.discover", lambda _cfg: {
        "alpha": Path("/models/alpha"), "beta": Path("/models/beta"),
    })

    class Workers:
        def snapshot(self):
            return []

        def handles(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(manager, "ModelWorkers", lambda *a, **k: Workers())
    with TestClient(manager.app) as test_client:
        yield test_client


def test_manager_lists_installed_model_names(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["alpha", "beta"]


def test_manager_info_reports_no_loaded_workers(client):
    response = client.get("/mlxh/info")
    assert response.json()["manager"] is True
    assert response.json()["model"] is None
    assert response.json()["workers"] == []


def test_manager_info_reports_image_capability_for_selected_model(client, monkeypatch):
    monkeypatch.setattr("mlxh.cli.model_supports_images", lambda path: path.endswith("alpha"))
    response = client.get("/mlxh/info", params={"model": "alpha"})
    assert response.json()["capabilities"] == {"images": True}


def test_json_api_requires_an_installed_model(client):
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    response = client.post("/v1/chat/completions", json={"model": "missing"})
    assert response.status_code == 404


def test_private_chat_transport_rejects_non_loopback_clients(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(manager, "CFG", {
        "worker_idle_timeout_s": 300, "models_dir": "/models",
    })
    monkeypatch.setattr("mlxh.cli.discover", lambda _cfg: {"alpha": Path("/models/alpha")})

    class Workers:
        def snapshot(self):
            return []

        def handles(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(manager, "ModelWorkers", lambda *a, **k: Workers())
    with TestClient(manager.app, client=("203.0.113.4", 1234)) as test_client:
        response = test_client.post("/mlxh/generate", json={"model": "alpha"})
    assert response.status_code == 403


def test_json_request_model_resolution():
    from starlette.requests import Request

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers": [(b"content-type", b"application/json")],
             "query_string": b"", "server": ("test", 80), "client": ("test", 1),
             "scheme": "http", "http_version": "1.1"}
    request = Request(scope)
    manager.MODELS = {"alpha": "/models/alpha"}
    assert manager._model_from_request(request, json.dumps({"model": "alpha"}).encode()) == "alpha"


def test_request_model_routes_to_its_worker_and_holds_lease(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(manager, "CFG", {
        "worker_idle_timeout_s": 300, "models_dir": "/models",
    })
    monkeypatch.setattr("mlxh.cli.discover", lambda _cfg: {
        "beta": Path("/models/beta"),
    })

    seen = []

    class Workers:
        def snapshot(self):
            return []

        def handles(self):
            return {}

        def close(self):
            pass

        @contextmanager
        def lease(self, model):
            assert model == "beta"
            yield SimpleNamespace(url="http://127.0.0.1:43210")

    monkeypatch.setattr(manager, "ModelWorkers", lambda *a, **k: Workers())
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aiter_raw(self):
            yield b'{"ok":true}'

        async def aclose(self):
            pass

    class FakeHTTP:
        def build_request(self, method, url, **kwargs):
            return method, url, kwargs

        async def send(self, built, stream=True):
            _method, url, options = built
            seen.append((url, json.loads(options["content"])["model"]))
            return FakeResponse()

        async def aclose(self):
            pass

    with TestClient(manager.app) as test_client:
        monkeypatch.setattr(manager, "HTTP", FakeHTTP())
        response = test_client.post("/v1/chat/completions", json={"model": "beta"})
    assert response.status_code == 200
    assert seen and seen[0] == ("http://127.0.0.1:43210/v1/chat/completions", "beta")


def test_private_sse_request_id_routes_cancel_to_same_worker(client, monkeypatch):
    tracker = manager._PrivateRequestTracker("beta")
    tracker.feed(b"event: start\ndata: {\"request_")
    tracker.feed(b"id\":\"req_1\"}\n\n")
    assert manager.ACTIVE_REQUESTS["req_1"] == "beta"

    seen = []

    class Workers:
        def close(self):
            pass

        @contextmanager
        def lease(self, model):
            assert model == "beta"
            yield SimpleNamespace(url="http://127.0.0.1:43210")

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        async def aiter_raw(self):
            yield b'{"cancelled":true}'

        async def aclose(self):
            pass

    class FakeHTTP:
        def build_request(self, method, url, **kwargs):
            return method, url, kwargs

        async def send(self, built, stream=True):
            _method, url, _options = built
            seen.append(url)
            return FakeResponse()

        async def aclose(self):
            pass

    monkeypatch.setattr(manager, "WORKERS", Workers())
    monkeypatch.setattr(manager, "HTTP", FakeHTTP())
    response = client.delete("/mlxh/requests/req_1")
    assert response.status_code == 200
    assert seen == ["http://127.0.0.1:43210/mlxh/requests/req_1"]
    tracker.close()
    assert "req_1" not in manager.ACTIVE_REQUESTS


def test_multipart_request_model_resolution():
    from starlette.requests import Request

    boundary = "abc123"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n"
            f"alpha\r\n--{boundary}--\r\n").encode()
    scope = {"type": "http", "method": "POST", "path": "/v1/images/edits",
             "headers": [(b"content-type", f"multipart/form-data; boundary={boundary}".encode())],
             "query_string": b"", "server": ("test", 80), "client": ("test", 1),
             "scheme": "http", "http_version": "1.1"}
    request = Request(scope)
    manager.MODELS = {"alpha": "/models/alpha"}
    assert manager._model_from_request(request, body) == "alpha"


def test_cancelled_request_releases_lease_after_worker_startup():
    started, finish_startup, released = threading.Event(), threading.Event(), threading.Event()

    class Lease:
        def __enter__(self):
            started.set()
            finish_startup.wait(timeout=2)
            return object()

        def __exit__(self, *_args):
            released.set()

    async def scenario():
        task = asyncio.create_task(manager._enter_lease(Lease()))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        finish_startup.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert released.is_set()


def test_stream_cleanup_is_shielded_from_response_cancellation():
    released = threading.Event()

    class Upstream:
        async def aclose(self):
            await asyncio.sleep(0.03)

    class Lease:
        def __exit__(self, *_args):
            released.set()

    async def scenario():
        task = asyncio.create_task(
            manager._close_upstream_and_release(Upstream(), Lease(), None))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert released.is_set()


def test_lease_released_when_client_disconnects_before_streaming():
    cleaned = []
    started = []

    async def body():
        started.append(True)
        yield b"never sent"

    async def cleanup():
        cleaned.append(True)

    response = manager._LeasedStreamingResponse(body(), cleanup=cleanup)

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        await asyncio.sleep(1)

    # spec 2.3 makes Starlette race the body against disconnect detection.
    scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
    asyncio.run(response(scope, receive, send))
    assert started == []
    assert cleaned == [True]


def test_model_name_read_from_json_with_other_content_type():
    from starlette.requests import Request

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers": [(b"content-type", b"text/plain;charset=UTF-8")],
             "query_string": b"", "server": ("test", 80), "client": ("test", 1),
             "scheme": "http", "http_version": "1.1"}
    manager.MODELS = {"alpha": "/models/alpha"}
    assert manager._model_from_request(Request(scope), b'{"model": "alpha"}') == "alpha"


def test_worker_start_waits_for_runtime_ready(monkeypatch, tmp_path):
    class Process:
        pid = 1234
        returncode = None

        def poll(self):
            return None

    class Response:
        status_code = 200

        def __init__(self, ready):
            self.ready = ready

        def json(self):
            return {"model": "alpha", "runtime": {"ready": self.ready}}

    class Client:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def get(self, _url):
            self.calls += 1
            return Response(self.calls >= 2)

    monkeypatch.setenv("MLXH_HOME", str(tmp_path))
    monkeypatch.setattr(manager, "MODELS", {"alpha": "/models/alpha"})
    monkeypatch.setattr(manager, "WORKERS", None)
    monkeypatch.setattr(manager, "_serve_argv", lambda *_args: ["fake-worker"])
    monkeypatch.setattr(manager.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    client = Client()
    monkeypatch.setattr(manager.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(manager.time, "sleep", lambda _seconds: None)

    handle = manager._start_worker("alpha")
    assert handle.info["runtime"]["ready"] is True
    assert client.calls == 2


def test_missing_image_runtime_gives_install_remediation(monkeypatch):
    manager.MODELS = {"image": "/models/image"}
    monkeypatch.setattr("mlxh.image_models.model_kind", lambda _path: "image")
    monkeypatch.setattr("mlxh.cli.serve_argv", lambda *_args: (_ for _ in ()).throw(SystemExit(1)))
    with pytest.raises(RuntimeError, match="mlxh images install"):
        manager._serve_argv("image", 45678)


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_manager_cli_rejects_non_finite_or_negative_idle_timeout(
        monkeypatch, capsys, value):
    monkeypatch.setattr("mlxh.cli.load_config", lambda: {
        "port": 1060, "host": "127.0.0.1",
    })
    monkeypatch.setattr(sys, "argv", ["mlxh-manager", "--worker-idle-timeout-s", value])
    with pytest.raises(SystemExit) as exc:
        manager.main()
    assert exc.value.code == 2
    assert "finite, non-negative" in capsys.readouterr().err


def test_real_worker_process_load_proxy_and_idle_unload(tmp_path, monkeypatch):
    from pathlib import Path

    model_path = tmp_path / "models" / "fake-model"
    model_path.mkdir(parents=True)
    monkeypatch.setenv("MLXH_HOME", str(tmp_path / "mlxh-home"))
    monkeypatch.setattr(manager, "CFG", {
        "worker_idle_timeout_s": 2, "models_dir": str(model_path.parent),
    })
    monkeypatch.setattr("mlxh.cli.discover", lambda _cfg: {
        "fake-model": Path(model_path),
    })
    child = r'''from http.server import BaseHTTPRequestHandler, HTTPServer
import json, sys
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"model": "fake-model", "model_kind": "language",
            "runtime": {"pid": __import__("os").getpid(), "ready": True,
                "requests": 1},
            "mlx": {"active_memory_bytes": 123}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        content_type = self.headers.get("Content-Type", "")
        if "application/json" in content_type:
            payload = json.loads(body)
        else:
            payload = {"contains_model": b'name="model"' in body,
                "contains_image": b'name="image"' in body,
                "body_bytes": len(body)}
        result = json.dumps({"path": self.path, "body": payload,
            "content_type": content_type}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(result))); self.end_headers(); self.wfile.write(result)
    def log_message(self, *_args): pass
HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
'''
    monkeypatch.setattr(manager, "_serve_argv", lambda _model, port: [
        sys.executable, "-c", child, str(port),
    ])

    with TestClient(manager.app) as test_client:
        for path in (
            "/v1/chat/completions", "/v1/responses", "/v1/messages",
            "/v1/messages/count_tokens", "/v1/images/generations",
        ):
            response = test_client.post(path, json={"model": "fake-model"})
            assert response.status_code == 200
            assert response.json()["path"] == path
            assert response.json()["body"] == {"model": "fake-model"}

        response = test_client.post(
            "/v1/images/edits", data={"model": "fake-model", "prompt": "edit"},
            files={"image": ("ref.png", b"fake-png-data", "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["path"] == "/v1/images/edits"
        assert response.json()["body"]["contains_model"] is True
        assert response.json()["body"]["contains_image"] is True
        loaded = test_client.get("/mlxh/info").json()["workers"]
        assert len(loaded) == 1
        assert loaded[0]["pid"] > 0
        assert loaded[0]["active_memory_bytes"] == 123
        assert loaded[0]["requests"] == 1

        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if not test_client.get("/mlxh/info").json()["workers"]:
                break
            time.sleep(0.1)
        assert test_client.get("/mlxh/info").json()["workers"] == []
