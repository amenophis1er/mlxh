"""Tool-call parsing (one format per model family) and user-tool loading.

mlxh ships no tools of its own. `chat --tools` loads them from
$MLXH_HOME/tools.py (default ~/.mlxh/tools.py), a user-owned module defining:

    TOOL_REGISTRY = {"tool_name": callable(**kwargs) -> dict}
    TOOL_SPECS    = [ ... OpenAI function-tool specs ... ]

A ready-made example (weather, time, calculator) ships in the repo at
examples/tools.py — copy it there to try tools out.

The parsing here is core, not demo: the API server uses it to translate a
model's tool calls into OpenAI `tool_calls` responses. Each family writes
calls its own way; the server picks a ToolFormat when it loads the model
(detect_format) and streams answer text only up to that format's marker.
"""

import importlib.util
import json
import os
import re
import uuid
from pathlib import Path

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S
)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
THINK_RE = re.compile(r"^\s*(<think>)?.*?</think>\s*", re.S)


def _call(name, args):
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _no_constant(name):
    raise ValueError(f"{name} is not JSON")  # NaN, Infinity: no client could read them back


_JSON = json.JSONDecoder(parse_constant=_no_constant)


def _skip_space(text, i):
    while i < len(text) and text[i].isspace():
        i += 1
    return i


class ToolFormat:
    """How one model family writes tool calls. parse returns (content, calls)."""
    name = ""
    marker = ""  # streamed text stops here: a call may be starting

    def parse(self, text):
        content, calls = self.split(text)
        return content.strip(), calls or None

    def split(self, text):
        """Read each call from its marker on. A marker with no call after it
        is prose and stays; a call that starts and cannot be read ends the
        search, so nothing inside a broken call is ever taken for a call."""
        kept, calls, at = [], [], 0
        while (start := text.find(self.marker, at)) != -1:
            try:
                call, end = self.read(text, start + len(self.marker))
            except (ValueError, IndexError, RecursionError):
                break
            if call is None:  # the marker was only mentioned
                kept.append(text[at:end])
            else:
                kept.append(text[at:start])
                calls.append(call)
            at = end
        kept.append(text[at:])
        return "".join(kept), calls

    def read(self, text, i):
        """The call that starts at text[i], as (call, index after it); (None,
        i) when no call starts there. Raises when one starts and is broken."""
        raise NotImplementedError

    def prepare_history(self, messages):
        """Messages as this family's chat template expects them."""
        return messages


class QwenXml(ToolFormat):
    """<tool_call><function=NAME><parameter=KEY>VALUE</parameter>…</function></tool_call>"""
    name, marker = "qwen_xml", "<tool_call>"

    def parse(self, text):
        # Raw output may still open with the template's think block.
        return super().parse(THINK_RE.sub("", text, count=1))

    def split(self, text):
        calls = []
        for name, body in TOOL_CALL_RE.findall(text):
            args = {}
            for key, value in PARAM_RE.findall(body):
                try:
                    args[key] = json.loads(value)
                except (ValueError, RecursionError):
                    args[key] = value
            calls.append(_call(name, args))
        return TOOL_CALL_RE.sub("", text), calls


class HermesJson(ToolFormat):
    """<tool_call>\n{"name": NAME, "arguments": {…}}\n</tool_call>"""
    name, marker = "hermes", "<tool_call>"
    _end = "</tool_call>"

    @staticmethod
    def _decode(text, i):
        """(value, index after it, extra braces skipped). Qwen2.5's template
        prints its example as {{"name": …}} and small models copy the brace."""
        for extra in range(3):
            try:
                return (*_JSON.raw_decode(text, i), extra)
            except ValueError:
                i = _skip_space(text, i + 1)
                if not text.startswith("{", i):
                    raise
        raise ValueError("not a JSON object")

    def read(self, text, i):
        i = _skip_space(text, i)
        if not text.startswith("{", i):
            return None, i
        value, i, extra = self._decode(text, i)
        args = value.get("arguments", value.get("parameters", {}))
        if isinstance(args, str):  # some models send arguments JSON-encoded
            args = _JSON.decode(args)
        if not isinstance(value.get("name"), str) or not isinstance(args, dict):
            raise ValueError("not a tool call")
        for _ in range(extra):  # the braces that closed the copied ones
            after = _skip_space(text, i)
            i = after + 1 if text.startswith("}", after) else i
        after = _skip_space(text, i)  # the closing tag is often left out
        return _call(value["name"], args), (
            after + len(self._end) if text.startswith(self._end, after) else i)


class Gemma4(ToolFormat):
    """<|tool_call>call:NAME{key:value,…}<tool_call|>, strings as <|"|>…<|"|>"""
    name, marker = "gemma4", "<|tool_call>"
    _end = "<tool_call|>"
    _QUOTE = '<|"|>'
    _HEAD = re.compile(r"\s*call:([\w.\-]+)\s*\{")
    _BARE = re.compile(r"[^,{}\[\]]+")
    _KEY = re.compile(r"[^:,{}\[\]]+")

    def read(self, text, i):
        head = self._HEAD.match(text, i)
        if not head:
            return None, i
        args, i = self._object(text, head.end() - 1)
        i = _skip_space(text, i)
        if not text.startswith(self._end, i):
            raise ValueError("call not closed")
        return _call(head.group(1), args), i + len(self._end)

    def _string(self, text, i):
        end = text.index(self._QUOTE, i + len(self._QUOTE))
        return text[i + len(self._QUOTE):end], end + len(self._QUOTE)

    def _value(self, text, i):
        i = _skip_space(text, i)
        if text.startswith(self._QUOTE, i):
            return self._string(text, i)
        if text[i] == "{":
            return self._object(text, i)
        if text[i] == "[":
            return self._items(text, i + 1, "]", lambda j: self._value(text, j))
        word = self._bare(self._BARE, text, i)
        try:
            return _JSON.decode(word.strip()), i + len(word)
        except ValueError:
            # the template writes null as None; any other bare word is text
            return (None if word.strip() == "None" else word.strip()), i + len(word)

    def _bare(self, pattern, text, i):
        found = pattern.match(text, i)
        if not found or "<|" in found.group() or "|>" in found.group():
            raise ValueError("not a bare word")  # a token of the format, out of place
        return found.group()

    def _pair(self, text, i):
        if text.startswith(self._QUOTE, i):
            key, i = self._string(text, i)
        else:
            word = self._bare(self._KEY, text, i)
            key, i = word.strip().strip('"'), i + len(word)
        i = _skip_space(text, i)
        if not key or text[i] != ":":
            raise ValueError("no key")
        value, i = self._value(text, i + 1)
        return (key, value), i

    def _object(self, text, i):
        pairs, i = self._items(text, i + 1, "}", lambda j: self._pair(text, j))
        return dict(pairs), i

    @staticmethod
    def _items(text, i, close, read):
        """Comma-separated items up to `close`; anything else between them is an error."""
        items = []
        while True:
            i = _skip_space(text, i)
            if text[i] == close:
                return items, i + 1
            item, i = read(i)
            items.append(item)
            i = _skip_space(text, i)
            if text[i] == ",":
                i += 1
            elif text[i] != close:
                raise ValueError("no separator")

    def prepare_history(self, messages):
        # Gemma's template writes an assistant message's text after its tool
        # responses and closes the turn there, leaving nothing open for the
        # model to answer in. Text that came before a call is dropped instead.
        return [{**m, "content": ""} if m.get("role") == "assistant" and m.get("tool_calls")
                else m for m in messages]


QWEN_XML, HERMES, GEMMA4 = QwenXml(), HermesJson(), Gemma4()
FORMATS = {f.name: f for f in (QWEN_XML, HERMES, GEMMA4)}


def format_for_template(template):
    """The format a chat template teaches its model; Qwen XML when unsure."""
    template = template or ""
    if "<|tool_call>" in template:
        return GEMMA4
    if "<function=" in template:
        return QWEN_XML
    if "<tool_call>" in template:
        return HERMES
    return QWEN_XML


def chat_template_text(model_dir):
    """The chat template a model directory uses, from the first place that
    has one (the order transformers reads them in), so a stale copy in
    tokenizer_config.json cannot outvote chat_template.jinja."""
    model_dir = Path(model_dir)
    jinja = sorted(model_dir.glob("chat_template*.jinja"))
    if jinja:
        return "\n".join(p.read_text(errors="replace") for p in jinja)
    for name in ("chat_template.json", "tokenizer_config.json"):
        try:
            value = json.loads((model_dir / name).read_text()).get("chat_template")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, list):  # named templates: [{"name":…, "template":…}]
            value = " ".join(t.get("template", "") for t in value if isinstance(t, dict))
        if isinstance(value, str) and value:
            return value
    return ""


def detect_format(model_dir):
    return format_for_template(chat_template_text(model_dir))


def parse_tool_calls(text, fmt=QWEN_XML):
    """Split model output into (content, openai_tool_calls_or_None)."""
    return fmt.parse(text)


def load_user_tools():
    """Load TOOL_REGISTRY/TOOL_SPECS from the user's tools file.

    Returns (registry, specs, path); registry is None when the file
    doesn't exist.
    """
    path = Path(os.environ.get("MLXH_HOME", Path.home() / ".mlxh")) / "tools.py"
    if not path.is_file():
        return None, None, path
    spec = importlib.util.spec_from_file_location("mlxh_user_tools", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "TOOL_REGISTRY", {}), getattr(mod, "TOOL_SPECS", []), path


def run_tool(registry, name, args):
    fn = registry.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}
    except Exception as e:
        return {"error": str(e)}
