"""Model-free API manager that starts isolated model workers on demand."""

from __future__ import annotations

import argparse
import asyncio
import json
import ipaddress
import logging
import os
import re
import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .model_workers import ModelWorkers

log = logging.getLogger("mlxh.manager")
CFG: dict = {}
MODELS: dict[str, str] = {}
WORKERS: ModelWorkers | None = None
HTTP: httpx.AsyncClient | None = None
ACTIVE_REQUESTS: dict[str, str] = {}
ACTIVE_REQUESTS_LOCK = threading.Lock()
MODELS_LOCK = threading.RLock()
# Callers of /mlxh/info give up after about 2 s, so worker probes must be faster.
INFO_PROBE_TIMEOUT_S = 1.0


@dataclass
class WorkerHandle:
    model: str
    port: int
    process: subprocess.Popen
    log_path: Path
    info: dict

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _refresh_models():
    global MODELS
    from .cli import discover

    found = {name: str(path.resolve()) for name, path in discover(CFG).items()}
    with MODELS_LOCK:
        # Publish a complete registry snapshot at once; requests may keep using
        # the previous dict while discovery constructs its replacement.
        MODELS = found


def _serve_argv(model: str, port: int) -> list[str]:
    from .cli import serve_argv
    from .image_models import model_kind

    path = MODELS[model]
    try:
        kind = model_kind(path)
    except ValueError as exc:
        raise RuntimeError(f"model '{model}' has unsupported metadata: {exc}") from None
    # Preserve the exact same settings as a fixed-model server, only bind the
    # private worker to loopback and an ephemeral port.
    try:
        return serve_argv(CFG, model, path, {"host": "127.0.0.1", "port": port})
    except SystemExit as exc:
        if kind == "image":
            raise RuntimeError("image runtime is unavailable; run `mlxh images install`") from None
        raise RuntimeError(f"could not prepare worker for '{model}'") from exc


def _start_worker(model: str) -> WorkerHandle:
    if model not in MODELS:
        raise KeyError(model)
    port = _free_port()
    resident = WORKERS.snapshot() if WORKERS is not None else []
    if resident:
        measured_peaks = [row["peak_memory_bytes"] for row in resident
                          if isinstance(row.get("peak_memory_bytes"), int)]
        estimate = sum(measured_peaks)
        note = (f" recorded worker peaks total about {estimate / 1e9:.1f} GB;"
                if estimate else " no worker peak estimate is available yet;")
        log.warning(
            "loading additional model %s while %d other worker(s) are resident;%s "
            "workers share unified memory and may infer concurrently",
            model, len(resident), note,
        )
    state_dir = Path(os.environ.get("MLXH_HOME", Path.home() / ".mlxh"))
    logs = state_dir / "workers"
    logs.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model)
    log_path = logs / f"{safe_name}.log"
    log_file = log_path.open("ab", buffering=0)
    try:
        argv = _serve_argv(model, port)
    except BaseException:
        log_file.close()
        raise
    try:
        from .worker_supervisor import supervisor_argv
        process = subprocess.Popen(
            supervisor_argv(argv, os.getpid()), stdin=subprocess.DEVNULL,
            stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
        )
    finally:
        log_file.close()
    handle = WorkerHandle(model, port, process, log_path, {})
    deadline = time.monotonic() + 900
    last_error = "worker did not become ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"worker exited ({process.returncode}); see {log_path}")
        try:
            with httpx.Client(timeout=1) as client:
                response = client.get(f"{handle.url}/mlxh/info")
            if response.status_code == 200:
                payload = response.json()
                if (payload.get("runtime") or {}).get("ready") is True:
                    handle.info = payload
                    return handle
                last_error = "worker model is still loading"
            else:
                last_error = f"worker returned HTTP {response.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    _stop_worker(handle)
    raise TimeoutError(f"worker startup timed out ({last_error}); see {log_path}")


def _stop_worker(handle: WorkerHandle):
    if handle.process.poll() is None:
        try:
            os.killpg(handle.process.pid, 15)
        except ProcessLookupError:
            pass
        try:
            handle.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(handle.process.pid, 9)
            except ProcessLookupError:
                pass
            handle.process.wait(timeout=3)


async def _enter_lease(lease):
    """Acquire a synchronous worker lease without leaking it on disconnect."""
    entering = asyncio.create_task(asyncio.to_thread(lease.__enter__))
    try:
        return await asyncio.shield(entering)
    except asyncio.CancelledError:
        # Cancelling asyncio.to_thread does not stop the underlying thread. Let
        # startup finish, then release the lease it may have acquired.
        try:
            await entering
        except Exception:
            pass
        else:
            await asyncio.to_thread(lease.__exit__, None, None, None)
        raise


async def _close_upstream_and_release(upstream, lease, tracker):
    """Finish stream cleanup even if Starlette cancels the response task."""
    # anyio re-delivers cancellation to every await inside a cancelled scope,
    # so asyncio.shield alone is not enough; shield the whole cleanup.
    with anyio.CancelScope(shield=True):
        try:
            await upstream.aclose()
        finally:
            if tracker is not None:
                tracker.close()
            await asyncio.to_thread(lease.__exit__, None, None, None)


class _LeasedStreamingResponse(StreamingResponse):
    """Release the worker lease when the response ends, however it ends.

    Starlette may cancel the response before the body generator runs even
    once (client already disconnected), in which case the generator's own
    ``finally`` never executes. Cleanup therefore lives here instead.
    """

    def __init__(self, content, *, cleanup, **kwargs):
        super().__init__(content, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup()


@asynccontextmanager
async def lifespan(_app):
    global WORKERS, HTTP
    _refresh_models()
    WORKERS = ModelWorkers(
        _start_worker, _stop_worker,
        idle_timeout_s=float(CFG.get("worker_idle_timeout_s", 300)),
    )
    HTTP = httpx.AsyncClient(timeout=None, follow_redirects=False)
    try:
        yield
    finally:
        WORKERS.close()
        await HTTP.aclose()
        HTTP = None


app = FastAPI(title="mlxh model manager", lifespan=lifespan)


@app.get("/mlxh/info")
async def info(request: Request):
    _refresh_models()
    worker_rows = WORKERS.snapshot() if WORKERS else []
    handles = WORKERS.handles() if WORKERS else {}
    if HTTP is not None:
        async def refresh(row):
            handle = handles.get(row["model"])
            if handle is None:
                return
            try:
                response = await HTTP.get(f"{handle.url}/mlxh/info",
                                          timeout=INFO_PROBE_TIMEOUT_S)
                response.raise_for_status()
                row.update(ModelWorkers._handle_diagnostics(response.json()))
            except (httpx.HTTPError, ValueError):
                row["state"] = "exited"

        await asyncio.gather(*(refresh(row) for row in worker_rows))
    result = {
        "model": None,
        "model_kind": "manager",
        "manager": True,
        "models": sorted(MODELS),
        "workers": worker_rows,
        "worker_idle_timeout_s": float(CFG.get("worker_idle_timeout_s", 300)),
    }
    model = request.query_params.get("model")
    models = MODELS
    if model in models:
        from .cli import model_supports_images
        result["capabilities"] = {"images": model_supports_images(models[model])}
    return result


@app.get("/v1/models")
def models():
    _refresh_models()
    models = MODELS
    return {"object": "list", "data": [
        {"id": name, "object": "model", "owned_by": "mlxh"}
        for name in sorted(models)
    ]}


def _model_from_request(request: Request, body: bytes) -> str:
    content_type = request.headers.get("content-type", "")
    model = None
    if "multipart/form-data" in content_type:
        from email.parser import BytesParser
        from email.policy import default

        message = BytesParser(policy=default).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
        )
        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == "model":
                model = part.get_content().strip()
                break
    elif body:
        # Workers parse JSON regardless of Content-Type, so the manager must too.
        try:
            data = json.loads(body)
        except (UnicodeError, json.JSONDecodeError):
            raise HTTPException(400, "invalid JSON request") from None
        if isinstance(data, dict):
            model = data.get("model")
    models = MODELS
    if isinstance(model, str) and model in models:
        return model
    if model is None:
        raise HTTPException(400, "request must include an installed model name in 'model'")
    raise HTTPException(404, f"no installed model named '{model}'")


class _PrivateRequestTracker:
    """Track request IDs emitted by the private chat SSE start event."""

    def __init__(self, model: str):
        self.model = model
        self.request_id = None
        self._buffer = ""

    def feed(self, chunk: bytes):
        self._buffer += chunk.decode("utf-8", errors="replace")
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            if "event: start" not in event:
                continue
            payload = "\n".join(
                line[5:].lstrip() for line in event.splitlines()
                if line.startswith("data:")
            )
            try:
                self.request_id = json.loads(payload).get("request_id")
            except (json.JSONDecodeError, AttributeError):
                continue
            if self.request_id:
                with ACTIVE_REQUESTS_LOCK:
                    ACTIVE_REQUESTS[self.request_id] = self.model

    def close(self):
        if self.request_id:
            with ACTIVE_REQUESTS_LOCK:
                if ACTIVE_REQUESTS.get(self.request_id) == self.model:
                    ACTIVE_REQUESTS.pop(self.request_id, None)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    if path in {"v1/models", "mlxh/info"}:
        raise HTTPException(404, "not found")
    if WORKERS is None:
        raise HTTPException(503, "model manager is starting")
    if path == "mlxh/generate" or path.startswith("mlxh/requests/"):
        host = request.client.host if request.client else ""
        try:
            is_local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_local = host == "testclient"
        if not is_local:
            raise HTTPException(403, "private chat transport is localhost-only")
    _refresh_models()
    body = await request.body()
    if len(body) > 52 * 1024 * 1024:
        raise HTTPException(413, "request body exceeds 52 MiB")
    cancel_match = re.fullmatch(r"mlxh/requests/([^/]+)", path)
    if request.method == "DELETE" and cancel_match:
        with ACTIVE_REQUESTS_LOCK:
            model = ACTIVE_REQUESTS.get(cancel_match.group(1))
        if model is None:
            raise HTTPException(404, "request not found")
    else:
        model = _model_from_request(request, body)
    try:
        lease = WORKERS.lease(model)
        worker = await _enter_lease(lease)
    except KeyError:
        raise HTTPException(404, f"no installed model named '{model}'") from None
    except Exception as exc:
        log.exception("failed to start worker for model %s", model)
        raise HTTPException(503, f"could not load model '{model}': {exc}") from None

    if HTTP is None:
        await asyncio.to_thread(lease.__exit__, None, None, None)
        raise HTTPException(503, "model manager is shutting down")

    url = f"{worker.url}/{path}"
    if request.url.query:
        url += "?" + request.url.query
    headers = {
        key: value for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection"}
    }
    try:
        upstream_request = HTTP.build_request(
            request.method, url, headers=headers, content=body,
        )
        upstream = await HTTP.send(upstream_request, stream=True)
    except asyncio.CancelledError:
        await asyncio.to_thread(lease.__exit__, None, None, None)
        raise
    except Exception:
        await asyncio.to_thread(lease.__exit__, None, None, None)
        raise HTTPException(502, f"worker for '{model}' is unavailable") from None

    response_headers = {
        key: value for key, value in upstream.headers.items()
        if key.lower() not in {"transfer-encoding", "connection"}
    }

    tracker = _PrivateRequestTracker(model) if path == "mlxh/generate" else None

    async def stream():
        async for chunk in upstream.aiter_raw():
            if tracker is not None:
                tracker.feed(chunk)
            yield chunk

    return _LeasedStreamingResponse(
        stream(), status_code=upstream.status_code,
        headers=response_headers,
        cleanup=lambda: _close_upstream_and_release(upstream, lease, tracker),
    )


def main():
    global CFG
    from .cli import load_config
    from .serve_app import SETTINGS

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--worker-idle-timeout-s", type=float, default=None)
    for arg, key in (("max-queued", "max_queued"), ("max-tokens-cap", "max_tokens_cap"),
                     ("memory-limit-gb", "memory_limit_gb"), ("cache-limit-gb", "cache_limit_gb"),
                     ("gen-timeout-s", "gen_timeout_s"), ("max-prompt-tokens", "max_prompt_tokens"),
                     ("prompt-cache", "prompt_cache"), ("thinking", "thinking"),
                     ("max-image-pixels", "max_image_pixels"), ("image-steps", "image_steps")):
        parser.add_argument("--" + arg, default=None)
    args = parser.parse_args()
    if args.worker_idle_timeout_s is not None:
        import math
        if not math.isfinite(args.worker_idle_timeout_s) or args.worker_idle_timeout_s < 0:
            parser.error("--worker-idle-timeout-s must be a finite, non-negative number")
    CFG = load_config()
    for key in SETTINGS:
        arg_value = getattr(args, key, None)
        if arg_value is not None:
            CFG[key] = arg_value
    if args.worker_idle_timeout_s is not None:
        CFG["worker_idle_timeout_s"] = args.worker_idle_timeout_s
    import uvicorn
    uvicorn.run(app, host=args.host or CFG["host"],
                port=args.port if args.port is not None else CFG["port"])


if __name__ == "__main__":
    main()
