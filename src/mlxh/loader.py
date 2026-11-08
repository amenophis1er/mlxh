"""Model loading abstraction.

Two families are supported behind one Runner interface:
- Prism Hadamard packs (model_type "prism_*" with a bundled runtime/ dir),
  loaded through the pack's own loader code.
- Stock MLX models (anything mlx_vlm or mlx_lm can load, e.g. mlx-community/*).
"""

import json
import os
import sys
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

        The block pool is sized in blocks, not bytes; max_bytes is accepted
        so every runner shares one enable_cache(max_bytes) contract.
        """
        from mlx_vlm.apc import APCManager
        self.apc = APCManager(num_blocks=num_blocks, block_size=block_size)
        return f"[apc] prompt cache enabled ({num_blocks * block_size // 1000}k-token block pool)"

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
        so the next agent turn (which echoes the assistant reply) hits. The
        LRU is bounded by bytes and entries; eviction happens on insert.
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

        The lookup excludes the last token: the model must see at least one
        token to produce logits, so an exact hit on the full prompt would be
        useless (mlx_lm raises on an empty prompt). Searching tokens[:-1]
        makes trimmable and non-trimmable caches behave the same way.
        """
        from mlx_lm.models.cache import make_prompt_cache
        cache, rest = self.prompt_cache.fetch_nearest_cache(self._cache_key, tokens[:-1])
        if cache is None:
            return make_prompt_cache(self.model), list(tokens), 0
        copied = sum(int(c.nbytes) for c in cache)
        # The copy lives outside the LRU: keep stored + in-flight under the cap.
        self.prompt_cache.trim_to(n_bytes=max(0, self.prompt_cache.max_bytes - copied))
        rest = list(rest) + tokens[-1:]
        return cache, rest, len(tokens) - len(rest)

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
        completed = False
        try:
            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=max_tokens, **kwargs
            ):
                response.cached_tokens = cached
                if cached:
                    # Report the full prompt and keep prompt_tps consistent
                    # with it (mlx-lm measured only the uncached suffix).
                    response.prompt_tokens = processed + cached
                    response.prompt_tps = (getattr(response, "prompt_tps", 0.0) or 0.0) * (
                        (processed + cached) / max(processed, 1))
                if key is not None:
                    key.append(response.token)
                yield response
            completed = True
        except GeneratorExit:
            # Cancelled by the caller: every yielded token is in the cache.
            completed = True
            raise
        finally:
            # Not stored after an error: the cache may be half-updated.
            if completed and key is not None and len(key) > processed + cached:
                self.prompt_cache.insert_cache(self._cache_key, key, cache)

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])


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
