"""How long a prompt each model may take.

A prompt costs memory for every token it holds in the model's cache, so the
server refuses one that is too long before processing it. The limit is the
model's own: its context length, or what the machine's memory leaves room
for, whichever is smaller. `max_prompt_tokens` overrides it for every model
(N), or turns it off (-1). Pure functions — no MLX, no FastAPI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

FALLBACK = 8192   # when a model tells us neither its context length nor its cost
FLOOR = 2048      # a model that barely fits still takes a short prompt
HEADROOM = 0.8    # share of the free memory a single prompt may plan on
# Tokenizers mark "no limit" with a huge number; no context window is this long.
SANE_CONTEXT = 100_000_000
CONTEXT_KEYS = ("max_position_embeddings", "model_max_length", "n_positions",
                "n_ctx", "max_seq_len", "context_length")
TEXT_CONFIGS = ("text_config", "llm_config", "language_config")


@dataclass
class PromptCap:
    tokens: int | None            # None: no cap
    source: str                   # setting | context | memory | fallback | off
    context_length: int | None = None
    memory_tokens: int | None = None

    def as_dict(self):
        return asdict(self)

    def describe(self):
        """Why the cap is what it is, for logs and error messages."""
        if self.source == "off":
            return "no limit (max_prompt_tokens is -1)"
        if self.source == "setting":
            return f"{self.tokens:,} tokens (max_prompt_tokens)"
        if self.source == "fallback":
            return f"{self.tokens:,} tokens (the model states no context length or cache size)"
        context = f"{self.context_length:,}" if self.context_length else "unknown"
        memory = f"~{self.memory_tokens:,}" if self.memory_tokens else "unknown"
        why = "its context length" if self.source == "context" else "what memory allows"
        return (f"{self.tokens:,} tokens, {why} "
                f"(context length {context}, memory allows {memory})")


def _sane(value):
    return (isinstance(value, int) and not isinstance(value, bool)
            and 0 < value < SANE_CONTEXT)


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def context_length(model_dir):
    """The model's context window in tokens, or None when it does not say.

    From config.json (the text model's own section too, never a vision or
    audio tower's), taking the smallest figure given; the tokenizer's limit
    only when the config has none."""
    config = _read_json(Path(model_dir) / "config.json")
    sections = [config] + [config[k] for k in TEXT_CONFIGS if isinstance(config.get(k), dict)]
    found = [s[k] for s in sections for k in CONTEXT_KEYS if _sane(s.get(k))]
    if found:
        return min(found)
    limit = _read_json(Path(model_dir) / "tokenizer_config.json").get("model_max_length")
    return limit if _sane(limit) else None


def _layers(cache):
    for layer in cache:
        inner = getattr(layer, "caches", None)
        if inner is None:
            yield layer
        else:
            yield from _layers(inner)


def cache_growth(cache):
    """(bytes each further token adds, bytes that stay fixed) for a prompt
    cache that has seen a few tokens. A key/value cache grows with the
    prompt; one with a window (max_size) stops at it; anything else (a
    recurrent state) is as big as it will get."""
    per_token = fixed = 0.0
    for layer in _layers(cache):
        shape = getattr(getattr(layer, "keys", None), "shape", None)
        if not shape or len(shape) < 3 or not shape[2]:
            fixed += layer.nbytes
            continue
        slot = layer.nbytes / shape[2]
        window = getattr(layer, "max_size", None)
        if window:
            fixed += slot * window
        else:
            per_token += slot
    return per_token, fixed


def memory_tokens(limit_bytes, active_bytes, cost):
    """Prompt tokens that fit in what the memory limit leaves free.

    cost: per_token and fixed bytes (cache_growth), how many copies of its
    cache a request holds at once, and bytes reserved for caches already
    stored. None when any of it is unknown."""
    if not limit_bytes or not cost or not cost.get("per_token"):
        return None
    copies = cost.get("copies", 1)
    free = (limit_bytes - active_bytes - cost.get("reserve", 0)) * HEADROOM
    return max(FLOOR, int((free - copies * cost.get("fixed", 0)) / (copies * cost["per_token"])))


def resolve(setting, context=None, memory=None):
    """The cap for one model. setting: max_prompt_tokens (0 auto, -1 off, N fixed)."""
    if setting < 0:
        return PromptCap(None, "off", context, memory)
    if setting > 0:
        return PromptCap(setting, "setting", context, memory)
    if context is None and memory is None:
        return PromptCap(FALLBACK, "fallback")
    if memory is None or (context is not None and context <= memory):
        return PromptCap(context, "context", context, memory)
    return PromptCap(memory, "memory", context, memory)
