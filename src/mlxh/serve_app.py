"""OpenAI-compatible REST API for any mlxh-managed model.

Invoked by `mlxh serve <model>`; can also run standalone:
    python serve_app.py --model-path /path/to/model --name mymodel --port 1060
"""

import argparse
import ipaddress
import json
import os
import queue
import sys
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import anthropic_compat as anth
from . import responses_compat as oresp
from .engine import (
    EngineError,
    GenerationRequest,
    GenerationTerminal,
    InferenceEngine,
    LogprobsOptions,
    capture_float32_logprobs as _capture_float32_logprobs,
    compute_logprobs as _compute_logprobs,
)
from .toolcalls import parse_tool_calls
from .images import (
    BODY_LIMIT as IMAGE_BODY_LIMIT, EDIT_BODY_LIMIT, MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_IMAGE_BYTES, ImageEditRequest, ImageError,
    parse_request as parse_image_request, stage_reference_image,
)

PRIVATE_BODY_LIMIT = 36 * 1024 * 1024


class _BodyTooLarge(Exception):
    pass


class PrivateBodyLimitMiddleware:
    """Reject oversized chat and Images bodies before FastAPI parses JSON."""

    def __init__(self, app, limit=PRIVATE_BODY_LIMIT):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        path = scope.get("path")
        limits = {
            "/mlxh/generate": (self.limit, "36 MiB"),
            "/v1/images/generations": (IMAGE_BODY_LIMIT, "256 KiB"),
            "/v1/images/edits": (EDIT_BODY_LIMIT, "50 MiB"),
        }
        if scope.get("type") != "http" or path not in limits:
            await self.app(scope, receive, send)
            return
        limit, limit_label = limits[path]
        is_image = path.startswith("/v1/images/")

        def oversized():
            if is_image:
                return JSONResponse(
                    ImageError(413, f"request body exceeds {limit_label}",
                               code="body_too_large").envelope(), 413,
                )
            return JSONResponse({"detail": f"request body exceeds {limit_label}"}, 413)
        headers = dict(scope.get("headers") or [])
        try:
            if int(headers.get(b"content-length", b"0")) > limit:
                await oversized()(scope, receive, send)
                return
        except ValueError:
            pass
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await oversized()(scope, receive, send)

@asynccontextmanager
async def lifespan(_app):
    try:
        yield
    finally:
        if getattr(engine, "model_kind", None) == "image":
            await run_in_threadpool(engine.stop)


app = FastAPI(title="mlxh", lifespan=lifespan)
app.add_middleware(PrivateBodyLimitMiddleware)
engine = None
MODEL_ID = "model"
SETTINGS = {
    "max_queued": 4,       # pending generations beyond the active one; more get a 503
    "max_tokens_cap": 16384,  # server-side ceiling on requested max_tokens; 0 = off
    "memory_limit_gb": 0.0,   # MLX unified-memory limit; 0 = auto (80% of RAM), -1 = off
    "cache_limit_gb": 0.0,    # MLX buffer-cache limit; 0 = off
    "gen_timeout_s": 600,     # hard stop for one generation; 0 = off
    "max_prompt_tokens": 8192,  # reject bigger prompts (KV cache = memory); 0 = off
    "prompt_cache": True,   # reuse KV blocks across requests (agents: huge TTFT win)
    "thinking": "auto",     # reasoning mode: auto (model default) / on / off
    "max_image_pixels": 4194304,
    "image_steps": 0,
}


@app.post("/v1/images/generations")
async def image_generations(request: Request):
    current_engine = engine
    try:
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise ImageError(400, "invalid JSON request") from None
        if current_engine is None:
            raise ImageError(503, "model engine is not configured", code="not_ready")
        if getattr(current_engine, "model_kind", "language") != "image":
            raise ImageError(400, "this model does not support image generation",
                             "model", "unsupported_model_operation")
        parsed = parse_image_request(body, current_engine.model_id, current_engine.settings)
        return await _wait_for_image_job(request, current_engine, parsed)
    except ImageError as exc:
        return JSONResponse(exc.envelope(), exc.status_code)


@app.post("/v1/images/edits")
async def image_edits(request: Request):
    current_engine = engine
    form = None
    parsed = None
    submitted = False
    try:
        if current_engine is None:
            raise ImageError(503, "model engine is not configured", code="not_ready")
        if getattr(current_engine, "model_kind", "language") != "image":
            raise ImageError(400, "this model does not support image editing",
                             "model", "unsupported_model_operation")
        if (current_engine.runner is not None
                and not getattr(current_engine.runner, "supports_edits", False)):
            raise ImageError(400, "the loaded model does not support image editing",
                             "model", "unsupported_model_operation")
        try:
            form = await request.form(max_files=MAX_REFERENCE_IMAGES,
                                      max_fields=12, max_part_size=64 * 1024)
        except HTTPException as exc:
            raise ImageError(400, "invalid multipart edit request",
                             code="invalid_multipart") from None
        values, uploads = _image_edit_form_values(form)
        staged = []
        try:
            for upload in uploads:
                if upload.size is not None and upload.size > MAX_REFERENCE_IMAGE_BYTES:
                    raise ImageError(400, "reference image exceeds the 25-MiB file limit",
                                     "image", "image_too_large")
                data = await upload.read(MAX_REFERENCE_IMAGE_BYTES + 1)
                if len(data) > MAX_REFERENCE_IMAGE_BYTES:
                    raise ImageError(400, "reference image exceeds the 25-MiB file limit",
                                     "image", "image_too_large")
                staged.append(stage_reference_image(data))
            parsed = parse_image_request(
                values, current_engine.model_id, current_engine.settings,
                source="openai-image-edits", input_images=staged,
                cleanup_input_images=True,
            )
        except BaseException:
            for path in staged:
                Path(path).unlink(missing_ok=True)
            raise
        finally:
            if form is not None:
                await form.close()
                form = None
        submitted = True
        return await _wait_for_image_job(request, current_engine, parsed)
    except ImageError as exc:
        return JSONResponse(exc.envelope(), exc.status_code)
    except EngineError:
        if parsed is not None:
            parsed.cleanup()
        exc = ImageError(503, "image engine unavailable or queue full", code="server_busy")
        return JSONResponse(exc.envelope(), 503)
    except Exception:
        if parsed is not None and not submitted:
            parsed.cleanup()
        raise
    finally:
        if form is not None:
            await form.close()


def _image_edit_form_values(form):
    allowed = {"model", "prompt", "n", "size", "seed", "steps", "quality",
               "response_format", "output_format", "output_compression", "user", "image"}
    values, uploads = {}, []
    for key, value in form.multi_items():
        if key not in allowed:
            raise ImageError(400, "unsupported edit request field", key, "unsupported_value")
        if key == "image":
            if not hasattr(value, "read") or not hasattr(value, "filename"):
                raise ImageError(400, "image must be an uploaded file", "image", "invalid_image")
            uploads.append(value)
            continue
        if key in values:
            raise ImageError(400, f"{key} must be supplied only once", key)
        if not isinstance(value, str):
            raise ImageError(400, f"{key} must be text", key)
        values[key] = value
    if not uploads:
        raise ImageError(400, "at least one reference image is required", "image")
    for key in ("n", "seed", "steps", "output_compression"):
        if key in values:
            try:
                values[key] = int(values[key])
            except ValueError:
                raise ImageError(400, f"{key} must be an integer", key) from None
    return values, uploads


async def _wait_for_image_job(request, current_engine, parsed):
    job = None
    completed = False
    try:
        job = current_engine.submit(parsed)
        created = int(time.time())
        while True:
            if await request.is_disconnected():
                return JSONResponse({}, 499)
            try:
                result = await run_in_threadpool(job.out.get, True, 0.1)
            except queue.Empty:
                continue
            completed = True
            if isinstance(result, ImageError):
                raise result
            return JSONResponse(await run_in_threadpool(result.response, created))
    except ImageError:
        raise
    except EngineError as exc:
        if job is None and isinstance(parsed, ImageEditRequest):
            parsed.cleanup()
        if exc.status_code == 400:
            error = ImageError(400, exc.detail, code="unsupported_model_operation")
            return JSONResponse(error.envelope(), 400)
        error = ImageError(503, "image engine unavailable or queue full", code="server_busy")
        return JSONResponse(error.envelope(), 503)
    finally:
        if job is not None and not completed:
            current_engine.cancel(job.request_id, expected_source=parsed.source)


def _request(body, source="openai"):
    options = body.get("_logprobs")
    logprobs = LogprobsOptions(**options) if options else None
    chat_template_kwargs = body.get("chat_template_kwargs") or {}
    thinking = (chat_template_kwargs.get("enable_thinking")
                if isinstance(chat_template_kwargs.get("enable_thinking"), bool)
                else None)
    return GenerationRequest(
        messages=body.get("messages", []), source=source,
        max_tokens=body.get("max_tokens") or body.get("max_completion_tokens") or 1024,
        tools=body.get("tools") or None,
        temperature=body.get("temperature"), top_p=body.get("top_p"),
        thinking=thinking, logprobs=logprobs,
    )


def submit(body, source="openai"):
    """Queue one normalized request on the process-wide inference engine."""
    if engine is None:
        raise HTTPException(503, "model engine is not configured")
    try:
        return engine.submit(_request(body, source)).out
    except EngineError as exc:
        raise HTTPException(exc.status_code, exc.detail) from None


def shape_logprobs(token_id, token_lp, top_ids, top_lps, decode, as_ids):
    """Build one OpenAI logprobs content entry from plain Python values."""
    def token_info(item_id, logprob):
        token = f"token_id:{item_id}" if as_ids else decode(item_id)
        return {
            "token": token,
            "logprob": float(logprob),
            "bytes": None if as_ids else list(token.encode("utf-8")),
        }

    return {
        **token_info(token_id, token_lp),
        "top_logprobs": [
            token_info(item_id, logprob)
            for item_id, logprob in zip(top_ids, top_lps)
        ],
    }


def _validate_id_list(body, field, *, nonempty=False):
    if field not in body:
        return []
    value = body[field]
    if not isinstance(value, list):
        raise HTTPException(400, f"{field} must be a list of integers")
    if nonempty and not value:
        raise HTTPException(400, f"{field} must not be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise HTTPException(400, f"{field} must contain only integers")
    if any(item < 0 for item in value):
        raise HTTPException(400, f"{field} must contain only non-negative ids")
    if len(set(value)) != len(value):
        raise HTTPException(400, f"{field} must not contain duplicate ids")
    return value


def _prepare_logprobs(body):
    """Validate logprobs options before the request reaches the MLX worker."""
    body.pop("_logprobs", None)
    if "logprobs" in body and not isinstance(body["logprobs"], bool):
        raise HTTPException(400, "logprobs must be a boolean")
    if not body.get("logprobs"):
        return
    if body.get("stream"):
        raise HTTPException(400, "logprobs are not available on streaming responses")
    top = body.get("top_logprobs", 0)
    if isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= 128:
        raise HTTPException(400, "top_logprobs must be an integer from 0 to 128")
    as_ids = body.get("return_tokens_as_token_ids", False)
    if not isinstance(as_ids, bool):
        raise HTTPException(400, "return_tokens_as_token_ids must be a boolean")
    ids = _validate_id_list(body, "logprob_token_ids", nonempty=True)
    allowed = _validate_id_list(body, "allowed_token_ids")
    body["_logprobs"] = {
        "top": top, "ids": ids, "allowed": allowed, "as_ids": as_ids,
    }


def _should_shape_logprobs(resp, previous_tokens):
    """Final stop records represent EOS, even when they flush buffered text."""
    return (int(resp.generation_tokens) > previous_tokens
            and resp.finish_reason != "stop")


MARKER = "<tool_call>"


def _public_finish(last, terminal):
    if terminal and terminal.outcome == "timed_out":
        return "length"
    if terminal:
        return terminal.finish_reason
    return getattr(last, "finish_reason", None) or "stop"


def _usage_counts(last, terminal):
    if terminal:
        return terminal.prompt_tokens, terminal.output_tokens
    return (int(getattr(last, "prompt_tokens", 0) or 0),
            int(getattr(last, "generation_tokens", 0) or 0))


@app.post("/v1/messages")
def anthropic_messages(body: dict):
    """Anthropic Messages API (what Claude Code speaks)."""
    oai = anth.to_openai_body(body)
    if os.environ.get("MLXH_DEBUG"):
        import pathlib
        dump = pathlib.Path(os.environ["MLXH_DEBUG"])
        with dump.open("a") as f:
            f.write(json.dumps({"anthropic": body, "openai": oai}) + "\n")
    chunks = submit(oai, "anthropic")
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    if body.get("stream"):
        def sse():
            ev = anth.sse_event
            parts, last, terminal, text_open, printed = [], None, None, False, 0
            # Emit message_start immediately and ping while the model chews on
            # the prompt: a silent connection during a long prompt-processing
            # phase makes agent clients assume the network died and retry,
            # piling duplicate jobs onto the queue.
            yield ev("message_start", {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "model": MODEL_ID, "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}}})
            while True:
                try:
                    item = chunks.get(timeout=15)
                except queue.Empty:
                    yield ev("ping", {"type": "ping"})
                    continue
                if item is None:
                    break
                if isinstance(item, GenerationTerminal):
                    terminal = item
                    continue
                if isinstance(item, BaseException):
                    yield ev("error", {"type": "error", "error": {
                        "type": "api_error", "message": str(item)}})
                    return
                last = item
                parts.append(item.text)
                full = "".join(parts)
                cut = full.find(MARKER)
                visible = full[:cut] if cut != -1 else full[: max(0, len(full) - len(MARKER))]
                if len(visible) > printed:
                    if not text_open:
                        yield ev("content_block_start", {
                            "type": "content_block_start", "index": 0,
                            "content_block": {"type": "text", "text": ""}})
                        text_open = True
                    yield ev("content_block_delta", {
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": visible[printed:]}})
                    printed = len(visible)
            full = "".join(parts)
            cut = full.find(MARKER)
            visible = full[:cut] if cut != -1 else full
            if len(visible) > printed:
                if not text_open:
                    yield ev("content_block_start", {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}})
                    text_open = True
                yield ev("content_block_delta", {
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": visible[printed:]}})
            if text_open:
                yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
            _, tool_calls = parse_tool_calls(full)
            index = 1
            for block in anth.content_blocks("", tool_calls):
                yield ev("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use", "id": block["id"],
                                      "name": block["name"], "input": {}}})
                yield ev("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": json.dumps(block["input"])}})
                yield ev("content_block_stop", {"type": "content_block_stop", "index": index})
                index += 1
            finish = _public_finish(last, terminal)
            prompt_tokens, output_tokens = _usage_counts(last, terminal)
            yield ev("message_delta", {"type": "message_delta",
                                       "delta": {"stop_reason": anth.stop_reason(tool_calls, finish),
                                                 "stop_sequence": None},
                                       "usage": {"input_tokens": prompt_tokens,
                                                 "output_tokens": output_tokens}})
            yield ev("message_stop", {"type": "message_stop"})

        return StreamingResponse(sse(), media_type="text/event-stream")

    parts, last, terminal = [], None, None
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, GenerationTerminal):
            terminal = item
            continue
        if isinstance(item, (HTTPException, EngineError)):
            if isinstance(item, EngineError):
                raise HTTPException(item.status_code, item.detail)
            raise item
        if isinstance(item, BaseException):
            raise HTTPException(500, str(item))
        parts.append(item.text)
        last = item
    text, tool_calls = parse_tool_calls("".join(parts))
    prompt_tokens, output_tokens = _usage_counts(last, terminal)
    return anth.message_response(
        MODEL_ID, text, tool_calls,
        prompt_tokens, output_tokens, _public_finish(last, terminal),
    )


@app.post("/v1/responses")
def openai_responses(body: dict):
    """OpenAI Responses API (what modern Codex speaks)."""
    oai = oresp.to_openai_body(body)
    chunks = submit(oai, "responses")
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"

    if body.get("stream"):
        def sse():
            ev = oresp.sse_event
            yield ev("response.created", {
                "type": "response.created",
                "response": oresp.response_object(resp_id, MODEL_ID, [],
                                                  oresp.usage_of(0, 0), "in_progress")})
            parts, last, terminal, err, text_open = [], None, None, None, False
            msg_item_id = f"msg_{uuid.uuid4().hex[:24]}"
            printed = 0
            while True:
                try:
                    item = chunks.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                if isinstance(item, GenerationTerminal):
                    terminal = item
                    continue
                if isinstance(item, BaseException):
                    err = item
                    break
                last = item
                parts.append(item.text)
                full = "".join(parts)
                cut = full.find(MARKER)
                visible = full[:cut] if cut != -1 else full[: max(0, len(full) - len(MARKER))]
                if len(visible) > printed:
                    if not text_open:
                        yield ev("response.output_item.added", {
                            "type": "response.output_item.added", "output_index": 0,
                            "item": {"id": msg_item_id, "type": "message",
                                     "role": "assistant", "status": "in_progress",
                                     "content": []}})
                        yield ev("response.content_part.added", {
                            "type": "response.content_part.added", "item_id": msg_item_id,
                            "output_index": 0, "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []}})
                        text_open = True
                    yield ev("response.output_text.delta", {
                        "type": "response.output_text.delta", "item_id": msg_item_id,
                        "output_index": 0, "content_index": 0,
                        "delta": visible[printed:]})
                    printed = len(visible)
            if err is not None:
                yield ev("response.failed", {
                    "type": "response.failed",
                    "response": {**oresp.response_object(resp_id, MODEL_ID, [],
                                                         oresp.usage_of(0, 0), "failed"),
                                 "error": {"code": "server_error", "message": str(err)}}})
                return
            full = "".join(parts)
            text, tool_calls = parse_tool_calls(full)
            if text_open:
                yield ev("response.output_text.done", {
                    "type": "response.output_text.done", "item_id": msg_item_id,
                    "output_index": 0, "content_index": 0, "text": text})
                yield ev("response.content_part.done", {
                    "type": "response.content_part.done", "item_id": msg_item_id,
                    "output_index": 0, "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": []}})
            output = oresp.output_items(text, tool_calls)
            for i, out_item in enumerate(output):
                yield ev("response.output_item.done", {
                    "type": "response.output_item.done", "output_index": i,
                    "item": out_item})
            prompt_tokens, output_tokens = _usage_counts(last, terminal)
            usage = oresp.usage_of(prompt_tokens, output_tokens)
            timed_out = terminal and terminal.outcome == "timed_out"
            response = oresp.response_object(
                resp_id, MODEL_ID, output, usage,
                "incomplete" if timed_out else "completed",
            )
            if timed_out:
                response["incomplete_details"] = {"reason": "max_output_tokens"}
            yield ev("response.incomplete" if timed_out else "response.completed", {
                "type": "response.incomplete" if timed_out else "response.completed",
                "response": response})

        return StreamingResponse(sse(), media_type="text/event-stream")

    parts, last, terminal = [], None, None
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, GenerationTerminal):
            terminal = item
            continue
        if isinstance(item, (HTTPException, EngineError)):
            if isinstance(item, EngineError):
                raise HTTPException(item.status_code, item.detail)
            raise item
        if isinstance(item, BaseException):
            raise HTTPException(500, str(item))
        parts.append(item.text)
        last = item
    text, tool_calls = parse_tool_calls("".join(parts))
    prompt_tokens, output_tokens = _usage_counts(last, terminal)
    usage = oresp.usage_of(prompt_tokens, output_tokens)
    timed_out = terminal and terminal.outcome == "timed_out"
    response = oresp.response_object(
        resp_id, MODEL_ID, oresp.output_items(text, tool_calls), usage,
        "incomplete" if timed_out else "completed",
    )
    if timed_out:
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    return response


@app.post("/v1/messages/count_tokens")
def anthropic_count_tokens(body: dict):
    if getattr(engine, "model_kind", None) == "image":
        raise HTTPException(400, "this model generates images and does not support chat")
    # Rough estimate; enough for agents that budget context with it.
    text = json.dumps(body.get("messages", [])) + json.dumps(body.get("system", ""))
    return {"input_tokens": max(1, len(text) // 4)}


@app.get("/mlxh/info")
def info():
    """Configuration and live diagnostics for this server process."""
    if engine is None:
        return {"model": MODEL_ID, "settings": SETTINGS}
    return engine.snapshot()


def _private_event(kind, payload):
    return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"


@app.post("/mlxh/generate")
async def private_generate(body: dict, request: Request):
    """Versioned localhost transport used by the terminal REPL."""
    host = request.client.host if request.client else ""
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host == "testclient"
    if not local:
        raise HTTPException(403, "private chat transport is localhost-only")
    if engine is None:
        raise HTTPException(503, "model engine is not configured")
    try:
        job = engine.submit(_request(body, "chat"))
    except EngineError as exc:
        raise HTTPException(exc.status_code, exc.detail) from None

    async def sse():
        terminal = None
        completed = False
        parts, printed = [], 0
        try:
            yield _private_event("start", {
                "request_id": job.request_id, "model": engine.model_id,
            })
            while True:
                if await request.is_disconnected():
                    engine.cancel(job.request_id)
                    return

                def get_one():
                    return job.out.get(timeout=0.25)

                try:
                    item = await run_in_threadpool(get_one)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                if isinstance(item, GenerationTerminal):
                    terminal = item
                    continue
                if isinstance(item, BaseException):
                    message = item.detail if isinstance(item, EngineError) else str(item)
                    yield _private_event("error", {"message": message})
                    completed = True
                    return
                phases = getattr(item, "phase_deltas", None)
                if phases is None:
                    phases = []
                    reasoning = getattr(item, "reasoning_text", "")
                    if reasoning:
                        phases.append(("reasoning", reasoning))
                    if item.text:
                        phases.append(("text", item.text))
                for phase, delta in phases:
                    if phase == "reasoning":
                        yield _private_event("reasoning_delta", {"text": delta})
                        continue
                    parts.append(delta)
                    full = "".join(parts)
                    cut = full.find(MARKER)
                    visible = (full[:cut] if cut != -1
                               else full[:max(0, len(full) - len(MARKER))])
                    if len(visible) > printed:
                        yield _private_event("text_delta", {"text": visible[printed:]})
                        printed = len(visible)

            full = "".join(parts)
            content, tool_calls = parse_tool_calls(full)
            if len(content) > printed:
                yield _private_event("text_delta", {"text": content[printed:]})
            if tool_calls:
                yield _private_event("tool_calls", {"calls": tool_calls})
            terminal = terminal or GenerationTerminal(
                "stop", "completed", 0, 0, 0.0, 0.0
            )
            yield _private_event("done", {
                "finish_reason": terminal.finish_reason,
                "outcome": terminal.outcome,
                "usage": {
                    "prompt_tokens": terminal.prompt_tokens,
                    "output_tokens": terminal.output_tokens,
                },
                "generation_s": terminal.generation_s,
                "tps": terminal.tps,
            })
            completed = True
        finally:
            if not completed:
                engine.cancel(job.request_id)

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.delete("/mlxh/requests/{request_id}")
def cancel_private_request(request_id: str, request: Request):
    host = request.client.host if request.client else ""
    try:
        local = ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = host == "testclient"
    if not local:
        raise HTTPException(403, "private chat transport is localhost-only")
    if engine is None or not engine.cancel(request_id):
        raise HTTPException(404, "request not found")
    return {"cancelled": True, "request_id": request_id}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "mlxh"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    _prepare_logprobs(body)

    def usage_of(last, terminal=None):
        prompt_tokens, output_tokens = _usage_counts(last, terminal)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        }

    chunks = submit(body)

    if body.get("stream"):
        def chunk_of(delta, finish=None, usage=None):
            payload = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage:
                payload["usage"] = usage
            return f"data: {json.dumps(payload)}\n\n"

        has_tools = bool(body.get("tools"))

        def sse():
            yield chunk_of({"role": "assistant"})
            last, terminal = None, None
            parts = []
            while True:
                try:
                    item = chunks.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                if isinstance(item, GenerationTerminal):
                    terminal = item
                    continue
                if isinstance(item, BaseException):
                    err = {"error": {"message": str(item), "type": "server_error"}}
                    yield f"data: {json.dumps(err)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                last = item
                if has_tools:
                    # Tool-call XML must be parsed whole; buffer instead of streaming.
                    parts.append(item.text)
                else:
                    yield chunk_of({"content": item.text})
            if has_tools:
                content, tool_calls = parse_tool_calls("".join(parts))
                if content:
                    yield chunk_of({"content": content})
                if tool_calls:
                    deltas = [
                        {"index": i, "id": tc["id"], "type": "function",
                         "function": tc["function"]}
                        for i, tc in enumerate(tool_calls)
                    ]
                    yield chunk_of({"tool_calls": deltas})
                finish = "tool_calls" if tool_calls else _public_finish(last, terminal)
            else:
                finish = _public_finish(last, terminal)
            yield chunk_of({}, finish=finish, usage=usage_of(last, terminal))
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    parts, last, terminal, logprobs_content = [], None, None, []
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, GenerationTerminal):
            terminal = item
            continue
        if isinstance(item, (HTTPException, EngineError)):
            if isinstance(item, EngineError):
                raise HTTPException(item.status_code, item.detail)
            raise item
        if isinstance(item, BaseException):
            raise HTTPException(500, str(item))
        parts.append(item.text)
        last = item
        if hasattr(item, "logprobs_out"):
            top_ids, top_lps, token_lp = item.logprobs_out
            logprobs_content.append(shape_logprobs(
                item.token, token_lp, top_ids, top_lps,
                engine.runner.decode_token, body["_logprobs"]["as_ids"],
            ))
    content, tool_calls = parse_tool_calls("".join(parts))
    message = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": completion_id, "object": "chat.completion",
        "created": created, "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": ({"content": logprobs_content}
                         if body.get("logprobs") else None),
            "finish_reason": ("tool_calls" if tool_calls
                              else _public_finish(last, terminal)),
        }],
        "usage": usage_of(last, terminal),
    }


def main():
    global MODEL_ID, engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--port", type=int, default=1060)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-queued", type=int, default=SETTINGS["max_queued"])
    ap.add_argument("--max-tokens-cap", type=int, default=SETTINGS["max_tokens_cap"])
    ap.add_argument("--memory-limit-gb", type=float, default=SETTINGS["memory_limit_gb"])
    ap.add_argument("--cache-limit-gb", type=float, default=SETTINGS["cache_limit_gb"])
    ap.add_argument("--gen-timeout-s", type=int, default=SETTINGS["gen_timeout_s"])
    ap.add_argument("--max-prompt-tokens", type=int, default=SETTINGS["max_prompt_tokens"])
    ap.add_argument("--prompt-cache", default=str(SETTINGS["prompt_cache"]))
    ap.add_argument("--thinking", choices=["auto", "on", "off"],
                    default=SETTINGS["thinking"])
    ap.add_argument("--max-image-pixels", type=int, default=4194304)
    ap.add_argument("--image-steps", type=int, default=0)
    args = ap.parse_args()
    SETTINGS.update(
        max_queued=args.max_queued, max_tokens_cap=args.max_tokens_cap,
        memory_limit_gb=args.memory_limit_gb, cache_limit_gb=args.cache_limit_gb,
        gen_timeout_s=args.gen_timeout_s, max_prompt_tokens=args.max_prompt_tokens,
        prompt_cache=str(args.prompt_cache).lower() in ("1", "true", "on", "yes"),
        thinking=args.thinking,
        max_image_pixels=args.max_image_pixels, image_steps=args.image_steps,
    )
    MODEL_ID = args.name or Path(args.model_path).name
    from .image_models import model_kind
    from .image_engine import ImageEngine
    try:
        kind = model_kind(args.model_path)
    except ValueError as exc:
        ap.error(f"unsupported model metadata: {exc}")
    engine_class = ImageEngine if kind == "image" else InferenceEngine
    if not 65536 <= args.max_image_pixels <= 4194304 or not 0 <= args.image_steps <= 100:
        ap.error("max-image-pixels must be 65536..4194304 and image-steps 0..100")
    engine = engine_class(
        args.model_path, MODEL_ID, SETTINGS, exit_on_load_failure=True
    )
    engine.start()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
