"""Inkling 2's message format, mapped onto what mlxh already speaks.

Inkling writes every turn as tagged messages rather than <think> blocks:

    <|message_model|><|content_thinking|>…<|end_message|>
    <|message_model|><|content_text|>…<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>

and stops on <|content_model_end_sampling|>. The parser here turns that into
the answer/reasoning split ReasoningParser gives, and a tool call into the
Qwen-style XML the server's parse_tool_calls already reads (every argument
JSON-encoded, so its values come back exactly as the model wrote them).
"""

from __future__ import annotations

import json
import re

MODEL_TYPES = {"inkling2_loop"}

_MARKER = re.compile(r"<\|[A-Za-z0-9_]{1,48}\|>")
_MARKER_START = re.compile(r"<\|[A-Za-z0-9_]{0,48}\|?")  # what a marker can begin as

_THINKING = "<|content_thinking|>"
_TEXT = "<|content_text|>"
_TOOL = "<|content_invoke_tool_json|>"
_END = "<|end_message|>"


def is_inkling(model_type: str) -> bool:
    return model_type in MODEL_TYPES


def is_inkling_model(path) -> bool:
    """A model directory whose config says it is Inkling 2."""
    from pathlib import Path
    try:
        return is_inkling(json.loads((Path(path) / "config.json").read_text()).get("model_type", ""))
    except (OSError, ValueError, AttributeError):
        return False


def _xml_safe_json(value) -> str:
    # "<" and ">" only occur inside JSON strings, where \u escapes mean the same:
    # a value holding "</parameter>" cannot close the XML around it
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def tool_call_xml(name: str, args: dict) -> str:
    params = "".join(
        f"<parameter={key}>\n{_xml_safe_json(value)}\n</parameter>\n"
        for key, value in args.items()
    )
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


_ENDS = (_END, "<|content_model_end_sampling|>", "<|endoftext|>")


def _is_control(marker, channel):
    """A marker that steers the format here. Inside a tool call's JSON only an end does."""
    if channel == "tool":
        return marker in (_END, "<|content_model_end_sampling|>")
    return marker in _ENDS or marker.startswith(("<|message_", "<|content_"))


class InklingParser:
    """Split Inkling's tagged output across arbitrary chunks (ReasoningParser's interface)."""

    def __init__(self):
        self.channel = None  # None (between messages: a header), "reasoning", "text", "tool"
        self.buffer = ""
        self.header = ""
        self.tool = ""
        self.events = []

    def _tool_text(self):
        raw, self.tool = self.tool, ""
        try:
            call = json.loads(raw)
            name = call.get("name") or self.header.strip()
            args = call.get("args", call.get("arguments", {}))
            if isinstance(name, str) and name and isinstance(args, dict):
                return tool_call_xml(name, args)
        except (json.JSONDecodeError, AttributeError):
            pass
        return raw  # not a call it could read: shown as it is, not lost

    def feed(self, text, final=False):
        self.buffer += text
        answer, reasoning, events = [], [], []

        def record(kind, chunk):
            if not chunk:
                return
            (reasoning if kind == "reasoning" else answer).append(chunk)
            if events and events[-1][0] == kind:
                events[-1] = (kind, events[-1][1] + chunk)
            else:
                events.append((kind, chunk))

        def content(chunk):
            if self.channel == "reasoning":
                record("reasoning", chunk)
            elif self.channel == "text":
                record("text", chunk)
            elif self.channel == "tool":
                self.tool += chunk
            else:
                self.header += chunk

        while self.buffer:
            start = self.buffer.find("<|")
            if start < 0:
                keep = 1 if not final and self.buffer.endswith("<") else 0  # "<" of a marker yet to come
                content(self.buffer[:len(self.buffer) - keep])
                self.buffer = self.buffer[len(self.buffer) - keep:]
                break
            if start:
                content(self.buffer[:start])
                self.buffer = self.buffer[start:]
            stop = self.buffer.find("|>")
            if stop < 0:
                if not final and _MARKER_START.fullmatch(self.buffer):
                    break  # a marker still arriving
                content(self.buffer[:2])  # "<|" in the text itself
                self.buffer = self.buffer[2:]
                continue
            if not _MARKER.fullmatch(self.buffer[:stop + 2]):
                content(self.buffer[:2])
                self.buffer = self.buffer[2:]
                continue
            marker, self.buffer = self.buffer[:stop + 2], self.buffer[stop + 2:]
            if not _is_control(marker, self.channel):
                content(marker)  # marker-shaped text: the model's words, kept
                continue
            if marker == _THINKING:
                self.channel = "reasoning"
            elif marker == _TEXT or (marker.startswith("<|content_") and marker not in (_TOOL, "<|content_model_end_sampling|>")):
                self.channel = "text"
            elif marker == _TOOL:
                self.channel, self.tool = "tool", ""
            elif marker == _END:
                if self.channel == "tool":
                    record("text", self._tool_text())
                self.channel, self.header = None, ""
            elif marker.startswith("<|message_"):
                self.channel, self.header = None, ""
            # any other marker (end of sampling, end of text) carries nothing to show

        if final and self.channel == "tool" and self.tool:
            record("text", self._tool_text())
        self.events = events
        return "".join(answer), "".join(reasoning)
