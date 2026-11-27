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


class ToolFormat:
    """How one model family writes tool calls. parse returns (content, calls)."""
    name = ""
    marker = ""  # streamed text stops here: a call may be starting

    def parse(self, text):
        text = THINK_RE.sub("", text, count=1)
        content, calls = self.split(text)
        return content.strip(), calls or None

    def split(self, text):
        raise NotImplementedError


class QwenXml(ToolFormat):
    """<tool_call><function=NAME><parameter=KEY>VALUE</parameter>…</function></tool_call>"""
    name, marker = "qwen_xml", "<tool_call>"

    def split(self, text):
        calls = []
        for name, body in TOOL_CALL_RE.findall(text):
            args = {}
            for key, value in PARAM_RE.findall(body):
                try:
                    args[key] = json.loads(value)
                except json.JSONDecodeError:
                    args[key] = value
            calls.append(_call(name, args))
        return TOOL_CALL_RE.sub("", text), calls


class HermesJson(ToolFormat):
    """<tool_call>\n{"name": NAME, "arguments": {…}}\n</tool_call>"""
    name, marker = "hermes", "<tool_call>"
    _end = "</tool_call>"

    @staticmethod
    def _decode(body):
        # Qwen2.5's template prints its example as {{"name": …}, and small
        # models copy the extra brace: skip leading ones until JSON decodes.
        body = body.lstrip()
        while body.startswith("{"):
            try:
                value, _ = json.JSONDecoder().raw_decode(body)
                return value
            except json.JSONDecodeError:
                body = body[1:].lstrip()
        return None

    @classmethod
    def _read(cls, body):
        value = cls._decode(body)
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            return None
        args = value.get("arguments", value.get("parameters", {}))
        if isinstance(args, str):  # some models send arguments JSON-encoded
            args = cls._decode(args)
        return _call(value["name"], args) if isinstance(args, dict) else None

    def split(self, text):
        kept, calls, at = [], [], 0
        while (start := text.find(self.marker, at)) != -1:
            body_at = start + len(self.marker)
            end = text.find(self._end, body_at)
            # a last call may stop at end of output without its closing tag
            stop = len(text) if end == -1 else end + len(self._end)
            call = self._read(text[body_at:len(text) if end == -1 else end])
            kept.append(text[at:start] if call else text[at:stop])
            if call:
                calls.append(call)
            at = stop
        kept.append(text[at:])
        return "".join(kept), calls


class Gemma4(ToolFormat):
    """<|tool_call>call:NAME{key:value,…}<tool_call|>, strings as <|"|>…<|"|>"""
    name, marker = "gemma4", "<|tool_call>"
    # a call's body never runs into the next call, so one broken call can't swallow it
    _CALL = re.compile(r"<\|tool_call>call:([\w.\-]+)(\{(?:(?!<\|tool_call>).)*?\})<tool_call\|>",
                       re.S)
    _QUOTE = '<|"|>'
    _BARE = re.compile(r"[^,:{}\[\]]+")

    def _value(self, text, i):
        if text.startswith(self._QUOTE, i):
            end = text.index(self._QUOTE, i + len(self._QUOTE))
            return text[i + len(self._QUOTE):end], end + len(self._QUOTE)
        if text[i] == "{":
            return self._object(text, i)
        if text[i] == "[":
            items, i = [], i + 1
            while text[i] != "]":
                item, i = self._value(text, i)
                items.append(item)
                i = i + 1 if text[i] == "," else i
            return items, i + 1
        word = self._BARE.match(text, i).group()
        try:
            return json.loads(word.strip()), i + len(word)
        except json.JSONDecodeError:
            return word.strip(), i + len(word)  # an unquoted word: keep it as text

    def _object(self, text, i):
        obj, i = {}, i + 1
        while text[i] != "}":
            colon = text.index(":", i)
            key = text[i:colon].strip()
            if key.startswith(self._QUOTE) and key.endswith(self._QUOTE):
                key = key[len(self._QUOTE):-len(self._QUOTE)]
            obj[key], i = self._value(text, colon + 1)
            i = i + 1 if text[i] == "," else i
        return obj, i + 1

    def _read(self, match):
        try:
            args, end = self._object(match.group(2), 0)
        except (ValueError, IndexError, AttributeError):
            return None
        return _call(match.group(1), args) if end == len(match.group(2)) else None

    def split(self, text):
        calls = []

        def take(match):
            call = self._read(match)
            if call is None:
                return match.group(0)
            calls.append(call)
            return ""

        return self._CALL.sub(take, text), calls


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
    """Every chat template a model directory carries, as one string."""
    model_dir = Path(model_dir)
    parts = [p.read_text(errors="replace") for p in sorted(model_dir.glob("chat_template*.jinja"))]
    for name in ("chat_template.json", "tokenizer_config.json"):
        try:
            value = json.loads((model_dir / name).read_text()).get("chat_template")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, list):  # named templates: [{"name":…, "template":…}]
            value = " ".join(t.get("template", "") for t in value if isinstance(t, dict))
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


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
