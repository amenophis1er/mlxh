"""OpenAI-compatible REST API for any mlxh-managed model.

Invoked by `mlxh serve <model>`; can also run standalone:
    python serve_app.py --model-path /path/to/model --name mymodel --port 1060
"""

import argparse
import base64
import binascii
import importlib.metadata
import json
import os
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import anthropic_compat as anth
from . import responses_compat as oresp
from .loader import load_runner
from .toolcalls import parse_tool_calls

app = FastAPI(title="mlxh")
runner = None
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
}
_STARTED = time.monotonic()
_stats = {
    "requests": 0,
    "tokens_generated": 0,
    "prompt_tokens": 0,
    "last_peak_memory_bytes": None,
    "busy": False,
}
_stats_lock = threading.Lock()


def _bump(key, n=1):
    with _stats_lock:
        _stats[key] += n


def _add(key, n):
    with _stats_lock:
        _stats[key] += n


def _set(key, value):
    with _stats_lock:
        _stats[key] = value

# All MLX work (model load AND generation) happens on one dedicated thread —
# MLX streams are per-thread state, so generating from FastAPI's threadpool
# crashes with "There is no Stream(cpu, 0) in current thread". The job queue
# also serializes requests, and a job always runs to completion (bounded by
# max_tokens) even if its client disconnects, so the server can't be wedged.
jobs = queue.Queue()
ready = threading.Event()


def gen_worker(model_path):
    global runner
    import mlx.core as mx
    from .cli import total_ram_bytes
    if SETTINGS["memory_limit_gb"] > 0:
        mx.set_memory_limit(int(SETTINGS["memory_limit_gb"] * 1e9))
        print(f"[mlx] memory limit {SETTINGS['memory_limit_gb']} GB", flush=True)
    elif SETTINGS["memory_limit_gb"] == 0:
        # Auto-cap so a runaway KV cache can't freeze the machine: better a
        # failed generation than an unresponsive Mac. -1 disables.
        ram = total_ram_bytes()
        if ram:
            limit = int(ram * 0.8)
            mx.set_memory_limit(limit)
            print(f"[mlx] memory limit {limit / 1e9:.0f} GB "
                  "(auto 80% of RAM; memory_limit_gb overrides, -1 disables)", flush=True)
    if SETTINGS["cache_limit_gb"] > 0:
        mx.set_cache_limit(int(SETTINGS["cache_limit_gb"] * 1e9))
        print(f"[mlx] cache limit {SETTINGS['cache_limit_gb']} GB", flush=True)
    print(f"Loading {MODEL_ID} from {model_path}...", flush=True)
    t0 = time.perf_counter()
    try:
        runner = load_runner(model_path)
    except Exception as e:
        print(e, flush=True)
        os._exit(1)
    if SETTINGS["prompt_cache"] and getattr(runner, "supports_cache", False):
        runner.enable_cache()
        print("[apc] prompt cache enabled (65k-token block pool)", flush=True)
    try:
        _set("last_peak_memory_bytes", mx.get_peak_memory())
        mx.reset_peak_memory()
    except Exception as stats_err:
        print(f"[mlx] stats error: {stats_err}", flush=True)
    print(f"Ready in {time.perf_counter() - t0:.1f}s "
          f"(images: {'yes' if runner.supports_images else 'no'})", flush=True)
    ready.set()
    timeout = SETTINGS["gen_timeout_s"]
    while True:
        job = jobs.get()
        out = job["out"]
        last = None
        stream = run_generation(job["body"])
        _set("busy", True)
        try:
            t0 = time.perf_counter()
            for resp in stream:
                last = resp
                out.put(resp)
                if timeout and time.perf_counter() - t0 > timeout:
                    print(f"[gen] timeout after {timeout}s, stopping", flush=True)
                    break
            out.put(None)
        except BaseException as e:
            out.put(e)
        finally:
            try:
                stream.close()
            finally:
                _set("busy", False)
                try:
                    with _stats_lock:
                        _stats["last_peak_memory_bytes"] = mx.get_peak_memory()
                        _stats["tokens_generated"] += (
                            last.generation_tokens if last else 0
                        )
                        _stats["prompt_tokens"] += last.prompt_tokens if last else 0
                    mx.reset_peak_memory()
                except Exception as stats_err:
                    print(f"[mlx] stats error: {stats_err}", flush=True)


def submit(body):
    """Queue a generation; returns the queue its chunks arrive on."""
    if not ready.wait(timeout=300):
        raise HTTPException(503, "model is still loading")
    if jobs.qsize() >= SETTINGS["max_queued"]:
        raise HTTPException(503, "Server busy: too many queued generations")
    out = queue.Queue()
    _bump("requests")
    jobs.put({"body": body, "out": out})
    return out


def extract_messages(raw_messages):
    """Normalize OpenAI messages: pull image parts out, keep text content."""
    messages, images, tmp_files = [], [], []
    for m in raw_messages:
        content = m.get("content", "")
        if isinstance(content, list):
            texts = []
            for part in content:
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    if not runner.supports_images:
                        raise HTTPException(400, "this model does not support images")
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        try:
                            payload = base64.b64decode(url.split(",", 1)[1])
                        except (IndexError, binascii.Error):
                            raise HTTPException(400, "Malformed data: image URL")
                        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        f.write(payload)
                        f.close()
                        images.append(f.name)
                        tmp_files.append(f.name)
                    elif Path(url).is_file():
                        images.append(url)
                    else:
                        raise HTTPException(
                            400, "image_url must be a data: URL or a local file path"
                        )
            content = "\n".join(texts)
        msg = {"role": m.get("role", "user"), "content": content or ""}
        if m.get("tool_calls"):
            # Template wants arguments as a dict; OpenAI sends a JSON string.
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        raise HTTPException(400, "tool_calls arguments must be JSON")
                calls.append({"function": fn})
            msg["tool_calls"] = calls
        messages.append(msg)
    return messages, images, tmp_files


def run_generation(body):
    """Yield generation chunks. Runs only on the gen_worker thread."""
    messages, images, tmp_files = extract_messages(body.get("messages", []))
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 1024
    if SETTINGS["max_tokens_cap"]:
        max_tokens = min(max_tokens, SETTINGS["max_tokens_cap"])
    tools = body.get("tools") or None
    thinking = {"on": True, "off": False}.get(SETTINGS["thinking"])
    prompt = runner.template(messages, num_images=len(images), tools=tools,
                             thinking=thinking)
    cap = SETTINGS["max_prompt_tokens"]
    if cap:
        n_prompt = len(prompt) if isinstance(prompt, (list, tuple)) else len(prompt) // 4
        if n_prompt > cap:
            raise HTTPException(
                400,
                f"prompt is ~{n_prompt} tokens, over max_prompt_tokens={cap}. "
                f"Long contexts grow the KV cache and can exhaust unified memory. "
                f"Raise deliberately: `mlxh config max_prompt_tokens {min(65536, max(cap * 2, n_prompt + 4096))}` "
                f"(or serve --max-prompt-tokens N).")
    print(f"[gen] start: {len(messages)} msgs, {len(images)} images, "
          f"{len(tools or [])} tools, max_tokens={max_tokens}", flush=True)
    t0 = time.perf_counter()
    last = None
    # When the template pre-opens a reasoning block, the model's output starts
    # mid-<think>; hold it back so clients only see the actual answer.
    pending = "" if (isinstance(prompt, str)
                     and prompt.rstrip().endswith("<think>")) else None
    try:
        for resp in runner.stream(
            prompt, images=images, max_tokens=max_tokens,
            temperature=body.get("temperature"), top_p=body.get("top_p"),
        ):
            last = resp
            if pending is not None:
                pending += resp.text
                if "</think>" not in pending:
                    continue
                import dataclasses
                after = pending.split("</think>", 1)[1].lstrip("\n")
                pending = None
                resp = dataclasses.replace(resp, text=after)
            yield resp
    finally:
        for f in tmp_files:
            Path(f).unlink(missing_ok=True)
        done = last.generation_tokens if last else 0
        print(f"[gen] end: {done} tokens in {time.perf_counter() - t0:.1f}s", flush=True)


MARKER = "<tool_call>"


@app.post("/v1/messages")
def anthropic_messages(body: dict):
    """Anthropic Messages API (what Claude Code speaks)."""
    oai = anth.to_openai_body(body)
    if os.environ.get("MLXH_DEBUG"):
        import pathlib
        dump = pathlib.Path(os.environ["MLXH_DEBUG"])
        with dump.open("a") as f:
            f.write(json.dumps({"anthropic": body, "openai": oai}) + "\n")
    chunks = submit(oai)
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    if body.get("stream"):
        def sse():
            ev = anth.sse_event
            parts, last, text_open, printed = [], None, False, 0
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
            finish = last.finish_reason if last else "stop"
            yield ev("message_delta", {"type": "message_delta",
                                       "delta": {"stop_reason": anth.stop_reason(tool_calls, finish),
                                                 "stop_sequence": None},
                                       "usage": {"input_tokens": last.prompt_tokens if last else 0,
                                                 "output_tokens": last.generation_tokens if last else 0}})
            yield ev("message_stop", {"type": "message_stop"})

        return StreamingResponse(sse(), media_type="text/event-stream")

    parts, last = [], None
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, HTTPException):
            raise item
        if isinstance(item, BaseException):
            raise HTTPException(500, str(item))
        parts.append(item.text)
        last = item
    text, tool_calls = parse_tool_calls("".join(parts))
    return anth.message_response(
        MODEL_ID, text, tool_calls,
        last.prompt_tokens if last else 0,
        last.generation_tokens if last else 0,
        last.finish_reason if last else "stop",
    )


@app.post("/v1/responses")
def openai_responses(body: dict):
    """OpenAI Responses API (what modern Codex speaks)."""
    oai = oresp.to_openai_body(body)
    chunks = submit(oai)
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"

    def drain():
        parts, last = [], None
        while True:
            item = chunks.get()
            if item is None:
                return "".join(parts), last, None
            if isinstance(item, BaseException):
                return "".join(parts), last, item
            parts.append(item.text)
            last = item
            yield item

    if body.get("stream"):
        def sse():
            ev = oresp.sse_event
            yield ev("response.created", {
                "type": "response.created",
                "response": oresp.response_object(resp_id, MODEL_ID, [],
                                                  oresp.usage_of(0, 0), "in_progress")})
            parts, last, err, text_open = [], None, None, False
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
            usage = oresp.usage_of(last.prompt_tokens if last else 0,
                                   last.generation_tokens if last else 0)
            yield ev("response.completed", {
                "type": "response.completed",
                "response": oresp.response_object(resp_id, MODEL_ID, output, usage)})

        return StreamingResponse(sse(), media_type="text/event-stream")

    parts, last = [], None
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, HTTPException):
            raise item
        if isinstance(item, BaseException):
            raise HTTPException(500, str(item))
        parts.append(item.text)
        last = item
    text, tool_calls = parse_tool_calls("".join(parts))
    usage = oresp.usage_of(last.prompt_tokens if last else 0,
                           last.generation_tokens if last else 0)
    return oresp.response_object(resp_id, MODEL_ID,
                                 oresp.output_items(text, tool_calls), usage)


@app.post("/v1/messages/count_tokens")
def anthropic_count_tokens(body: dict):
    # Rough estimate; enough for agents that budget context with it.
    text = json.dumps(body.get("messages", [])) + json.dumps(body.get("system", ""))
    return {"input_tokens": max(1, len(text) // 4)}


@app.get("/mlxh/info")
def info():
    """Configuration and live diagnostics for this server process."""
    with _stats_lock:
        stats = dict(_stats)
    try:
        import mlx.core as mx
        mlx_stats = {
            "active_memory_bytes": int(mx.get_active_memory()),
            "cache_memory_bytes": int(mx.get_cache_memory()),
            "last_peak_memory_bytes": stats["last_peak_memory_bytes"],
        }
        mlx_version = importlib.metadata.version("mlx")
    except Exception:
        mlx_stats = {
            "active_memory_bytes": None,
            "cache_memory_bytes": None,
            "last_peak_memory_bytes": None,
        }
        mlx_version = None
    return {
        "model": MODEL_ID,
        "settings": SETTINGS,
        "mlx": mlx_stats,
        "runtime": {
            "uptime_s": int(time.monotonic() - _STARTED),
            "pid": os.getpid(),
            "ready": ready.is_set(),
            "busy": stats["busy"],
            "queue_depth": jobs.qsize(),
            "requests": stats["requests"],
            "prompt_tokens": stats["prompt_tokens"],
            "tokens_generated": stats["tokens_generated"],
            "mlx_version": mlx_version,
        },
    }


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

    def usage_of(last):
        return {
            "prompt_tokens": last.prompt_tokens,
            "completion_tokens": last.generation_tokens,
            "total_tokens": last.prompt_tokens + last.generation_tokens,
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
            last = None
            parts = []
            while True:
                try:
                    item = chunks.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
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
                finish = "tool_calls" if tool_calls else (last.finish_reason or "stop")
            else:
                finish = last.finish_reason or "stop"
            yield chunk_of({}, finish=finish, usage=usage_of(last))
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    parts, last = [], None
    while True:
        item = chunks.get()
        if item is None:
            break
        if isinstance(item, HTTPException):
            raise item
        if isinstance(item, BaseException):
            raise HTTPException(500, str(item))
        parts.append(item.text)
        last = item
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
            "finish_reason": "tool_calls" if tool_calls else (last.finish_reason or "stop"),
        }],
        "usage": usage_of(last),
    }


def main():
    global MODEL_ID
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
    args = ap.parse_args()
    SETTINGS.update(
        max_queued=args.max_queued, max_tokens_cap=args.max_tokens_cap,
        memory_limit_gb=args.memory_limit_gb, cache_limit_gb=args.cache_limit_gb,
        gen_timeout_s=args.gen_timeout_s, max_prompt_tokens=args.max_prompt_tokens,
        prompt_cache=str(args.prompt_cache).lower() in ("1", "true", "on", "yes"),
        thinking=args.thinking,
    )
    MODEL_ID = args.name or Path(args.model_path).name
    threading.Thread(target=gen_worker, args=(args.model_path,), daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
