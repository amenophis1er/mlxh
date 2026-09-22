"""OpenAI Responses API translation (the subset Codex uses).

Modern Codex only speaks `wire_api = "responses"`. Requests are translated
into the internal OpenAI-chat-style body the server already understands;
results are wrapped back into Responses objects / SSE events.
Pure functions — no FastAPI, no model code.
"""

import json
import time
import uuid


def _content_to_parts(content):
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        kind = block.get("type", "")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": block.get("text", "")})
        elif kind == "input_image":
            url = block.get("image_url", "")
            if isinstance(url, dict):
                url = url.get("url", "")
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def to_openai_body(body):
    """Responses API request -> internal chat-style request body."""
    messages = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    items = body.get("input", [])
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    for item in items:
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if role == "system" and messages and messages[0]["role"] == "system":
                role = "user"  # keep templates happy: one leading system max
            content = _content_to_parts(item.get("content"))
            messages.append({"role": role, "content": content})
        elif kind == "function_call":
            messages.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": item.get("call_id", ""), "type": "function",
                "function": {"name": item.get("name", ""),
                             "arguments": item.get("arguments") or "{}"}}]})
        elif kind == "function_call_output":
            out = item.get("output", "")
            messages.append({"role": "tool",
                             "content": out if isinstance(out, str) else json.dumps(out)})
        # reasoning / other item types: skipped

    tools = [
        {"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {}),
        }}
        for t in body.get("tools") or [] if t.get("type") == "function"
    ]

    out = {"messages": messages,
           "max_tokens": body.get("max_output_tokens") or body.get("max_tokens") or 4096}
    if tools:
        out["tools"] = tools
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            out[key] = body[key]
    return out


def output_items(text, tool_calls):
    items = []
    if text:
        items.append({
            "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message",
            "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    for tc in tool_calls or []:
        items.append({
            "id": f"fc_{uuid.uuid4().hex[:24]}", "type": "function_call",
            "status": "completed",
            "call_id": tc["id"],
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
        })
    return items


def response_object(resp_id, model_id, output, usage, status="completed"):
    return {
        "id": resp_id, "object": "response", "created_at": int(time.time()),
        "status": status, "model": model_id, "output": output,
        "usage": usage, "error": None, "incomplete_details": None,
    }


def usage_of(prompt_tokens, output_tokens):
    return {"input_tokens": prompt_tokens, "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens}


def sse_event(kind, data):
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"
