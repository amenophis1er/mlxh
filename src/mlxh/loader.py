"""Model loading abstraction.

Two families are supported behind one Runner interface:
- Prism Hadamard packs (model_type "prism_*" with a bundled runtime/ dir),
  loaded through the pack's own loader code.
- Stock MLX models (anything mlx_vlm or mlx_lm can load, e.g. mlx-community/*).
"""

import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Must be set before transformers is first imported: silences import-time
# advisories (e.g. "PyTorch was not found") the user can't act on.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Gemma-style audio towers trip a harmless mel-filter warning while their
# (unused) audio preprocessor initializes; keep startup clean.
warnings.filterwarnings("ignore", message=".*mel filter.*", category=UserWarning)


class VLMRunner:
    supports_images = True
    supports_cache = True

    def __init__(self, model, processor, config):
        self.model, self.processor, self.config = model, processor, config
        self.apc = None

    def enable_cache(self, max_bytes=None, num_blocks=4096, block_size=16):
        """Automatic prefix caching: reuse KV blocks across requests.

        mlx-vlm sizes its pool in blocks, not bytes, so prompt_cache_gb does
        not apply here; the banner says so instead of silently ignoring it.
        """
        from mlx_vlm.apc import APCManager
        self.apc = APCManager(num_blocks=num_blocks, block_size=block_size)
        return (f"[apc] prompt cache enabled ({num_blocks * block_size // 1000}k-token "
                "block pool; prompt_cache_gb does not apply to mlx-vlm models)")

    def template(self, messages, num_images=0, tools=None, thinking=None):
        from mlx_vlm.prompt_utils import apply_chat_template
        kwargs = {"tools": tools} if tools else {}
        if thinking is not None:
            kwargs["enable_thinking"] = thinking
        return apply_chat_template(
            self.processor, self.config, messages, num_images=num_images, **kwargs
        )

    def stream(self, prompt, images=None, max_tokens=1024, temperature=None,
               top_p=None, logits_processors=None):
        from mlx_vlm import stream_generate
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if logits_processors:
            kwargs["logits_processors"] = logits_processors
        if self.apc is not None:
            kwargs["apc_manager"] = self.apc
        yield from stream_generate(
            self.model, self.processor, prompt,
            image=images or None, max_tokens=max_tokens, **kwargs,
        )

    def decode_token(self, token_id: int) -> str:
        return self.processor.tokenizer.decode([token_id])


class TextRunner:
    supports_images = False
    supports_cache = True

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.prompt_cache = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.cached_tokens_total = 0

    def enable_cache(self, max_bytes, max_entries=32):
        """Prefix caching: keep finished KV caches, reuse the longest prefix.

        Mirrors mlx_lm.server: entries are keyed by prompt + generated tokens
        so the next agent turn (which echoes the assistant reply) hits. A
        cache that cannot be trimmed also keeps a snapshot taken at the end of
        the prompt (see stream). The LRU is bounded by bytes and entries;
        eviction happens on insert.
        """
        from mlx_lm.models.cache import LRUPromptCache
        self.prompt_cache = LRUPromptCache(max_size=max_entries, max_bytes=int(max_bytes))
        self._cache_key = "model"  # trie key; nn.Module itself is not hashable
        return f"[cache] prefix cache enabled (cap {max_bytes / 1e9:.1f} GB, prompt_cache_gb overrides)"

    def cache_stats(self):
        if self.prompt_cache is None:
            return None
        return {
            "entries": len(self.prompt_cache),
            "bytes": int(self.prompt_cache.nbytes),
            "max_bytes": int(self.prompt_cache.max_bytes),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "cached_tokens_total": self.cached_tokens_total,
        }

    def _fetch_cache(self, tokens):
        """Return (cache, tokens_to_process, cached_count) for this prompt.

        The model must see at least one token to produce logits (mlx_lm
        raises on an empty prompt), so an exact hit steps back one token when
        the cache can be trimmed. A copy that would leave nothing cached is
        discarded rather than charged against the cap.
        """
        from mlx_lm.models.cache import (
            can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache,
        )
        cache, rest = self.prompt_cache.fetch_nearest_cache(self._cache_key, tokens)
        if cache is not None and not rest and can_trim_prompt_cache(cache):
            trim_prompt_cache(cache, 1)
            rest = tokens[-1:]
        cached = len(tokens) - len(rest)
        if cache is None or cached <= 0 or not rest:
            return make_prompt_cache(self.model), list(tokens), 0
        return cache, list(rest), cached

    def template(self, messages, num_images=0, tools=None, thinking=None):
        kwargs = {"tools": tools} if tools else {}
        if thinking is not None:
            kwargs["enable_thinking"] = thinking
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **kwargs
        )

    def stream(self, prompt, images=None, max_tokens=1024, temperature=None,
               top_p=None, logits_processors=None):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        kwargs = {}
        if temperature is not None or top_p is not None:
            kwargs["sampler"] = make_sampler(
                temp=temperature if temperature is not None else 1.0,
                top_p=top_p if top_p is not None else 1.0,
            )
        if logits_processors:
            kwargs["logits_processors"] = logits_processors

        cache, key, cached = None, None, 0
        if self.prompt_cache is not None and not isinstance(prompt, str):
            tokens = list(prompt)
            cache, prompt, cached = self._fetch_cache(tokens)
            kwargs["prompt_cache"] = cache
            key = list(tokens)
            if cached:
                self.cache_hits += 1
                self.cached_tokens_total += cached
            else:
                self.cache_misses += 1
        processed = len(prompt)
        snapshot, prefill_s = None, 0.0
        completed = False
        try:
            if key is not None and self._wants_snapshot(cache, prompt):
                started = time.perf_counter()
                snapshot = (key[:-1], self._prefill_snapshot(cache, prompt[:-1]))
                prefill_s = time.perf_counter() - started
                prompt = prompt[-1:]
            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=max_tokens, **kwargs
            ):
                response.cached_tokens = cached
                # Usage counts the full prompt; prompt_tps stays what was
                # measured over the tokens actually processed.
                response.prompt_tokens = processed + cached
                if prefill_s and response.prompt_tps:
                    response.prompt_tps = processed / (prefill_s + len(prompt) / response.prompt_tps)
                if key is not None:
                    key.append(response.token)
                yield response
            completed = True
        except GeneratorExit:
            # Cancelled by the caller. The last step was scheduled lazily;
            # force it now so a failure surfaces here, not in a later request.
            completed = self._cache_is_sound(cache)
            raise
        finally:
            # The LRU enforces the byte cap on insert (superseded prefixes are
            # dropped first), so a failed request never evicts anything.
            if completed and snapshot is not None:
                self._drop_superseded(snapshot[0])
            if completed and key is not None and len(key) > processed + cached:
                self.prompt_cache.insert_cache(self._cache_key, key, cache)
            if completed and snapshot is not None:
                self.prompt_cache.insert_cache(self._cache_key, *snapshot, cache_type="user")

    @staticmethod
    def _wants_snapshot(cache, prompt):
        """A trimmable cache is cut back to any prefix of the final entry; one
        that cannot be (linear attention keeps a running state) is only reusable
        up to where it was stored. The next turn's history drops the reasoning,
        so it shares the prompt with this one, not the reply."""
        from mlx_lm.models.cache import can_trim_prompt_cache
        return len(prompt) > 1 and not can_trim_prompt_cache(cache)

    def _prefill_snapshot(self, cache, tokens, step=2048):
        """Process all but the last prompt token, as mlx_lm's generate_step
        does (same chunks, same stream: a miss computes exactly what it would
        uncached), and return a copy of the cache at that point."""
        import mlx.core as mx
        from mlx_lm.generate import generation_stream
        with mx.stream(generation_stream):
            for start in range(0, len(tokens), step):
                self.model(mx.array(tokens[start:start + step])[None], cache=cache)
                mx.eval([c.state for c in cache])
                mx.clear_cache()
        return copy.deepcopy(cache)

    def _drop_superseded(self, key):
        """Before a snapshot at `key` is stored, drop what it makes dead weight:
        the conversation's earlier snapshots (strict prefixes of key) and the
        replies stored after them that the conversation did not keep (they
        extend a dropped prefix but not key). Untrimmable entries are never
        dropped by the LRU itself, so without this each turn would be charged
        again for the whole history before it. Uses LRUPromptCache's internals,
        which both pinned mlx-lm versions share."""
        lru = self.prompt_cache
        stored = [(kind, tokens) for kind, queue in lru._lru._lrus.items()
                  for _, tokens in queue]

        def extends(tokens, prefix):
            return len(tokens) > len(prefix) and tokens[:len(prefix)] == prefix

        prefixes = [tokens for _, tokens in stored if extends(key, tokens)]
        stale = [tokens for kind, tokens in stored
                 if kind == "assistant" and tokens not in prefixes
                 and tokens[:len(key)] != key
                 and any(extends(tokens, prefix) for prefix in prefixes)]
        for tokens in prefixes + stale:
            entry = lru._trie.pop(self._cache_key, tokens)
            lru._n_bytes -= entry.nbytes
            lru._n_bytes_by_type[entry.cache_type] -= entry.nbytes
            lru._lru.remove(self._cache_key, tokens)

    @staticmethod
    def _cache_is_sound(cache):
        try:
            import mlx.core as mx
            mx.eval([c.state for c in cache])
            return True
        except Exception:
            return False

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])


class InklingRunner(TextRunner):
    """Inkling 2 (mlx-lm's inkling2 model): its own effort setting and message format."""

    def template(self, messages, num_images=0, tools=None, thinking=None):
        kwargs = {"tools": tools} if tools else {}
        if thinking is False:
            kwargs["reasoning_effort"] = "none"
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **kwargs
        )

    def output_parser(self):
        from .inkling import InklingParser
        return InklingParser()


def _quiet_libraries():
    # Load-time advisories the user can't act on (unknown custom model_type,
    # tokenizer-regex heuristics); the pack runtimes own those choices.
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass


def load_runner(model_dir):
    _quiet_libraries()
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    model_type = cfg.get("model_type", "")

    if model_type.startswith("prism_") and (model_dir / "runtime").is_dir():
        sys.path.insert(0, str(model_dir / "runtime"))
        if cfg.get("components", {}).get("vision"):
            from vision_artifact import chat_config, load_vl_model
            model, processor, config = load_vl_model(model_dir)
            return VLMRunner(model, processor, chat_config(config))
        from artifact import load_model
        from mlx_lm.utils import load_tokenizer
        model, _ = load_model(model_dir)
        return TextRunner(model, load_tokenizer(model_dir))

    from .inkling import is_inkling
    if is_inkling(model_type):
        # mlx-vlm does not know it (and says so loudly): mlx-lm's own model, text in, text out
        from mlx_lm import load as lm_load
        model, tokenizer = lm_load(str(model_dir))
        return InklingRunner(model, tokenizer)

    def _short(e, n=180):
        s = " ".join(str(e).split())
        return s[:n] + ("…" if len(s) > n else "")

    try:
        from mlx_vlm import load as vlm_load
        from mlx_vlm.utils import load_config
        model, processor = vlm_load(str(model_dir))
        return VLMRunner(model, processor, load_config(str(model_dir)))
    except Exception as vlm_err:
        try:
            from mlx_lm import load as lm_load
            model, tokenizer = lm_load(str(model_dir))
            return TextRunner(model, tokenizer)
        except Exception as lm_err:
            raise RuntimeError(
                f"cannot load '{model_dir.name}' with the installed MLX stack\n"
                f"  mlx-vlm: {_short(vlm_err)}\n"
                f"  mlx-lm:  {_short(lm_err)}\n"
                f"  -> the model may need a newer mlx-vlm/mlx-lm; try updating mlxh"
            ) from None
