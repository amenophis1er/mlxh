"""Pinned MFLUX adapter. No optional imports until a worker loads a model."""

from .image_models import MFLUX_VERSION, image_metadata
from .images import (
    EFFECTIVE_REFERENCE_PIXELS, MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_IMAGE_BYTES, MAX_REFERENCE_PIXELS,
)


class ImageRunner:
    supports_images = False  # image INPUT; diffusion produces images
    supports_edits = False
    supports_cache = False
    default_width = default_height = 1024
    default_steps = 4
    min_dimension = 256
    max_dimension = 2048
    dimension_multiple = 16

    def __init__(self, path):
        from importlib.metadata import version, PackageNotFoundError

        try:
            installed = version("mflux")
        except PackageNotFoundError:
            installed = None
        if installed != MFLUX_VERSION:
            raise RuntimeError("image runtime needs repair; run: mlxh images install")
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux.variants.txt2img.flux import Flux1

        self.metadata = image_metadata(path)
        if not self.metadata:
            raise ValueError("unsupported image model")
        family = self.metadata["family"]
        if family == "flux1-schnell":
            model_class = Flux1
        elif family == "flux2-klein-4b":
            from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit

            class _Flux2KleinEditCompat(Flux2KleinEdit):
                """Keep txt2img parity while loading Klein weights only once.

                In MFLUX 0.20.0, the edit class's no-reference path reaches
                concatenate(None). Dispatch that path to Klein's txt2img
                predictor; reference edits keep using the edit predictor.
                """
                def _predict(self, transformer):
                    edit_predict = super()._predict(transformer)
                    text_predict = Flux2Klein._predict(transformer)

                    def predict(**kwargs):
                        if kwargs.get("image_latents") is None:
                            kwargs.pop("image_latents", None)
                            kwargs.pop("image_latent_ids", None)
                            kwargs.pop("kv_cache", None)
                            kwargs.pop("negative_kv_cache", None)
                            return text_predict(**kwargs)
                        return edit_predict(**kwargs)

                    return predict

            model_class = _Flux2KleinEditCompat
        elif family == "qwen-image":
            from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
            model_class = QwenImage
        else:
            raise ValueError(f"unsupported image family: {family}")
        self.model_config = getattr(ModelConfig, self.metadata["model_config"])()
        self.model = model_class(model_path=str(path), model_config=self.model_config)
        self.supports_edits = bool(self.metadata.get("supports_edits"))
        self.edit_limits = ({
            "max_reference_images": MAX_REFERENCE_IMAGES,
            "max_reference_image_bytes": MAX_REFERENCE_IMAGE_BYTES,
            "max_reference_pixels": MAX_REFERENCE_PIXELS,
            "effective_reference_pixels": EFFECTIVE_REFERENCE_PIXELS,
            "preprocessing": "aspect-preserving resize then center-crop to multiples of 16",
        } if self.supports_edits else {})
        if self.model.bits != self.metadata["quantization_bits"]:
            raise ValueError("image weights do not match declared quantization")
        tokenizers = self.model.tokenizers
        tokenizer_key = {"flux1-schnell": "t5", "flux2-klein-4b": "qwen3",
                         "qwen-image": "qwen"}[family]
        self.prompt_tokenizer = tokenizers[tokenizer_key]
        self.t5_limit = (self.prompt_tokenizer.max_length
                         if family == "flux1-schnell" else None)
        self.prompt_limit = (self.t5_limit or self.model_config.max_sequence_length
                             or self.prompt_tokenizer.max_length)
        self.default_width = self.default_height = 1024
        from mflux.cli.defaults.defaults import model_inference_steps
        step_model = {"flux1-schnell": "schnell"}.get(family, family)
        self.default_steps = model_inference_steps(step_model)
        self.prompt_limits = {
            "tokenizer": tokenizer_key,
            "tokens": self.prompt_limit,
            "enforcement": "hard",
        }
        if family == "flux1-schnell":
            self.prompt_limits.update({
                "t5_tokens": self.t5_limit,
                "clip_tokens": 77,
                "clip_enforcement": "advisory",
                "clip_overflow": "truncated_by_model",
            })

    def validate_prompt(self, prompt):
        from .images import ImageError

        tokenizer = self.prompt_tokenizer
        raw_tokenizer = tokenizer.tokenizer
        formatted_prompt = prompt
        if getattr(tokenizer, "template", None):
            formatted_prompt = tokenizer.template.format(prompt)
        elif getattr(tokenizer, "use_chat_template", False):
            formatted_prompt = raw_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
                **getattr(tokenizer, "chat_template_kwargs", {}),
            )
        options = {"truncation": False, "padding": False}
        if hasattr(tokenizer, "add_special_tokens"):
            options["add_special_tokens"] = tokenizer.add_special_tokens
        ids = raw_tokenizer(formatted_prompt, **options)["input_ids"]
        limit = self.prompt_limit
        if limit and len(ids) > limit:
            label = "T5" if self.t5_limit else self.metadata["family"]
            raise ImageError(400, f"prompt exceeds {label}'s {limit}-token limit",
                             "prompt", "prompt_too_long")

    def generate(self, request, steps, check):
        import mlx.core as mx
        from mflux.callbacks.callback_registry import CallbackRegistry

        class Progress:
            def call_before_loop(self, **kwargs):
                check(0)

            def call_in_loop(self, t, latents, **kwargs):
                # MFLUX calls subscribers before its lazy eval. Evaluate here so
                # cancellation cannot leave a queued GPU graph behind.
                mx.eval(latents)
                check(int(t) + 1)

            def call_after_loop(self, **kwargs):
                check(steps)

        self.model.callbacks = CallbackRegistry()
        self.model.callbacks.register(Progress())
        try:
            check(0)
            result = self.model.generate_image(
                prompt=request.prompt, seed=request.seed, width=request.width,
                height=request.height, num_inference_steps=steps,
                **({"image_paths": None} if self.supports_edits else {}),
            )
            check(steps)
            return result.image
        finally:
            self.model.callbacks = CallbackRegistry()
            # MFLUX caches prompts/embeddings by default; do not retain user input.
            self.model.prompt_cache.clear()

    def edit(self, request, steps, check):
        if not self.supports_edits:
            from .images import ImageError
            raise ImageError(400, "this model does not support image editing",
                             "model", "unsupported_model_operation")
        import mlx.core as mx
        from mflux.callbacks.callback_registry import CallbackRegistry

        class Progress:
            def call_before_loop(self, **kwargs):
                check(0)

            def call_in_loop(self, t, latents, **kwargs):
                mx.eval(latents)
                check(int(t) + 1)

            def call_after_loop(self, **kwargs):
                check(steps)

        self.model.callbacks = CallbackRegistry()
        self.model.callbacks.register(Progress())
        try:
            check(0)
            result = self.model.generate_image(
                prompt=request.prompt, seed=request.seed, width=request.width,
                height=request.height, num_inference_steps=steps,
                image_paths=request.input_images,
            )
            check(steps)
            return result.image
        finally:
            self.model.callbacks = CallbackRegistry()
            self.model.prompt_cache.clear()
