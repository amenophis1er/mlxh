"""OpenAI-compatible REST API for any mlxh-managed model.

Invoked by `mlxh serve <model>`; can also run standalone:
    python serve_app.py --model-path /path/to/model --name mymodel --port 8081
"""

import argparse
import base64
import binascii
import json
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from loader import load_runner
from toolcalls import parse_tool_calls

app = FastAPI(title="mlxh")
runner = None
MODEL_ID = "model"
MAX_QUEUED = 4  # pending generations beyond the active one; more get a 503

# All MLX work (model load AND generation) happens on one dedicated thread —
# MLX streams are per-thread state, so generating from FastAPI's threadpool
# crashes with "There is no Stream(cpu, 0) in current thread". The job queue
# also serializes requests, and a job always runs to completion (bounded by
# max_tokens) even if its client disconnects, so the server can't be wedged.
jobs = queue.Queue()
ready = threading.Event()


def gen_worker(model_path):
    global runner
    print(f"Loading {MODEL_ID} from {model_path}...", flush=True)
    t0 = time.perf_counter()
    runner = load_runner(model_path)
    print(f"Ready in {time.perf_counter() - t0:.1f}s "
          f"(images: {'yes' if runner.supports_images else 'no'})", flush=True)
    ready.set()
    while True:
        job = jobs.get()
        out = job["out"]
        try:
            for resp in run_generation(job["body"]):
                out.put(resp)
            out.put(None)
        except BaseException as e:
            out.put(e)


def submit(body):
    """Queue a generation; returns the queue its chunks arrive on."""
    if not ready.wait(timeout=300):
        raise HTTPException(503, "model is still loading")
    if jobs.qsize() >= MAX_QUEUED:
        raise HTTPException(503, "Server busy: too many queued generations")
    out = queue.Queue()
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
    tools = body.get("tools") or None
    prompt = runner.template(messages, num_images=len(images), tools=tools)
    print(f"[gen] start: {len(messages)} msgs, {len(images)} images, "
          f"{len(tools or [])} tools, max_tokens={max_tokens}", flush=True)
    t0 = time.perf_counter()
    last = None
    try:
        for resp in runner.stream(
            prompt, images=images, max_tokens=max_tokens,
            temperature=body.get("temperature"), top_p=body.get("top_p"),
        ):
            last = resp
            yield resp
    finally:
        for f in tmp_files:
            Path(f).unlink(missing_ok=True)
        done = last.generation_tokens if last else 0
        print(f"[gen] end: {done} tokens in {time.perf_counter() - t0:.1f}s", flush=True)


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
                item = chunks.get()
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
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    MODEL_ID = args.name or Path(args.model_path).name
    threading.Thread(target=gen_worker, args=(args.model_path,), daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
