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

    def __init__(self, model, processor, config, context_window=None):
        self.model, self.processor, self.config = model, processor, config
        # Detected model context window (in tokens); None when unknown.
        self.context_window = context_window
        self.apc = None

    def enable_cache(self, num_blocks=4096, block_size=16):
        """Automatic prefix caching: reuse KV blocks across requests."""
        from mlx_vlm.apc import APCManager
        self.apc = APCManager(num_blocks=num_blocks, block_size=block_size)

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
    supports_cache = False  # text-only models: caching lands later

    def __init__(self, model, tokenizer, context_window=None):
        self.model, self.tokenizer = model, tokenizer
        self.context_window = context_window

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
        yield from stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=max_tokens, **kwargs
        )

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


_POSITION_KEYS = ("max_position_embeddings", "n_positions", "context_length")


def detect_context_window(config):
    """Best-effort context window (in tokens) from a HF-style config dict.

    Flat text models carry `max_position_embeddings` at the top level; most
    vision models and Prism packs nest the text model under `text_config`
    (older layouts use `language_config`). There is deliberately no rope_scaling
    math: scaled windows are model-specific, so a model whose config
    understates (or overstates) its practical window can be overridden with the
    `context_window` config key. Returns None when unknown.
    """
    scopes = [config, config.get("text_config"), config.get("language_config")]
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        for key in _POSITION_KEYS:
            value = scope.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def context_window_of(model_path):
    """Detected context window for a model directory, or None."""
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(config, dict):
        return None
    return detect_context_window(config)


def resolve_context_window(setting, detected):
    """Effective context window from the `context_window` setting.

    -1 disables window use entirely, 0 (default) uses the value detected
    from the model config, and a positive value overrides detection. None
    means no usable window is known. Shared by workers, the fixed-model
    server, and the manager so every surface reports the same limit.
    """
    override = setting or 0
    if override < 0:
        return None
    if override > 0:
        return override
    if isinstance(detected, int) and not isinstance(detected, bool) and detected > 0:
        return detected
    return None


def load_runner(model_dir):
    _quiet_libraries()
    model_dir = Path(model_dir)
    cfg_path = model_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    model_type = cfg.get("model_type", "")
    context_window = detect_context_window(cfg)

    if model_type.startswith("prism_") and (model_dir / "runtime").is_dir():
        sys.path.insert(0, str(model_dir / "runtime"))
        if cfg.get("components", {}).get("vision"):
            from vision_artifact import chat_config, load_vl_model
            model, processor, config = load_vl_model(model_dir)
            return VLMRunner(model, processor, chat_config(config), context_window)
        from artifact import load_model
        from mlx_lm.utils import load_tokenizer
        model, _ = load_model(model_dir)
        return TextRunner(model, load_tokenizer(model_dir), context_window)

    def _short(e, n=180):
        s = " ".join(str(e).split())
        return s[:n] + ("…" if len(s) > n else "")

    try:
        from mlx_vlm import load as vlm_load
        from mlx_vlm.utils import load_config
        model, processor = vlm_load(str(model_dir))
        return VLMRunner(model, processor, load_config(str(model_dir)), context_window)
    except Exception as vlm_err:
        try:
            from mlx_lm import load as lm_load
            model, tokenizer = lm_load(str(model_dir))
            return TextRunner(model, tokenizer, context_window)
        except Exception as lm_err:
            raise RuntimeError(
                f"cannot load '{model_dir.name}' with the installed MLX stack\n"
                f"  mlx-vlm: {_short(vlm_err)}\n"
                f"  mlx-lm:  {_short(lm_err)}\n"
                f"  -> the model may need a newer mlx-vlm/mlx-lm; try updating mlxh"
            ) from None
