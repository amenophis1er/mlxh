"""Anthropic Messages API translation (the subset coding agents use).

Requests are translated into the internal OpenAI-style body the server
already understands; results are wrapped back into Anthropic message
objects / SSE events. Pure functions — no FastAPI, no model code.
"""

import json
import uuid


def to_openai_body(body):
    """Anthropic /v1/messages request -> internal OpenAI-style request body."""
    messages = []

    system = body.get("system")
    if system:
        if isinstance(system, list):
            system = "\n".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            )
        messages.append({"role": "system", "content": system})

    for m in body.get("messages", []):
        role, content = m.get("role", "user"), m.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts, tool_calls = [], []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif kind == "image":
                src = block.get("source", {})
                if src.get("type") == "base64":
                    media = src.get("media_type", "image/png")
                    parts.append({"type": "image_url", "image_url": {
                        "url": f"data:{media};base64,{src.get('data', '')}"}})
            elif kind == "tool_use":
                tool_calls.append({"id": block.get("id", ""), "type": "function",
                                   "function": {"name": block.get("name", ""),
                                                "arguments": json.dumps(block.get("input", {}))}})
            elif kind == "tool_result":
                result = block.get("content", "")
                if isinstance(result, list):
                    result = "\n".join(
                        b.get("text", "") for b in result if b.get("type") == "text"
                    )
                messages.append({"role": "tool",
                                 "content": result if isinstance(result, str) else json.dumps(result)})
        if parts or tool_calls:
            msg = {"role": role, "content": parts or ""}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)

    tools = [
        {"type": "function", "function": {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }}
        for t in body.get("tools") or []
    ]

    # Coding agents send extra system-role messages mid-conversation
    # (system reminders); chat templates only allow system first. Demote any
    # later system message to a user message, preserving its position.
    for i, m in enumerate(messages):
        if i > 0 and m.get("role") == "system":
            m["role"] = "user"

    out = {"messages": messages, "max_tokens": body.get("max_tokens") or 1024}
    if tools:
        out["tools"] = tools
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            out[key] = body[key]
    return out


def stop_reason(tool_calls, finish_reason):
    if tool_calls:
        return "tool_use"
    return "max_tokens" if finish_reason == "length" else "end_turn"


def content_blocks(text, tool_calls):
    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in tool_calls or []:
        blocks.append({
            "type": "tool_use",
            "id": tc["id"].replace("call_", "toolu_"),
            "name": tc["function"]["name"],
            "input": json.loads(tc["function"]["arguments"]),
        })
    return blocks


def message_response(model_id, text, tool_calls, prompt_tokens, output_tokens, finish_reason):
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content_blocks(text, tool_calls),
        "stop_reason": stop_reason(tool_calls, finish_reason),
        "stop_sequence": None,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": output_tokens},
    }


def sse_event(kind, data):
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"
