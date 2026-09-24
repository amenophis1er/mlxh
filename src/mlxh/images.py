"""Typed image requests and strict OpenAI Images protocol validation."""

import base64
import io
import math
import re
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .engine import EngineError

BODY_LIMIT = 256 * 1024
OUTPUT_LIMIT = 32 * 1024 * 1024
EDIT_BODY_LIMIT = 50 * 1024 * 1024
MAX_REFERENCE_IMAGES = 4
MAX_REFERENCE_IMAGE_BYTES = 25 * 1024 * 1024
MAX_REFERENCE_PIXELS = 16_000_000
EFFECTIVE_REFERENCE_PIXELS = 1_048_576


class ImageError(EngineError):
    def __init__(self, status, detail, param=None, code="invalid_value"):
        super().__init__(status, detail)
        self.param, self.code = param, code

    def envelope(self):
        return {"error": {"message": self.detail,
                          "type": "invalid_request_error" if self.status_code < 500 else "server_error",
                          "param": self.param, "code": self.code}}


@dataclass
class ImageGenerationRequest:
    prompt: str
    width: int
    height: int
    output_format: str
    output_compression: int | None
    seed: int
    steps: int | None = None
    source: str = "openai-images"


@dataclass
class ImageEditRequest(ImageGenerationRequest):
    input_images: list[str] | None = None
    cleanup_input_images: bool = False

    def cleanup(self):
        if not self.cleanup_input_images:
            return
        for path in self.input_images or ():
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass


@dataclass
class ImageGenerationResult:
    image: bytes
    request: ImageGenerationRequest
    steps: int
    generation_s: float

    def response(self, created):
        r = self.request
        mlxh = {"seed": r.seed, "steps": self.steps,
                "generation_s": round(self.generation_s, 3)}
        if isinstance(r, ImageEditRequest):
            mlxh.update(operation="edit", reference_images=len(r.input_images or ()))
        return {"created": created, "data": [{"b64_json": base64.b64encode(self.image).decode()}],
                "output_format": r.output_format, "size": f"{r.width}x{r.height}",
                "quality": "auto", "mlxh": mlxh}


def parse_request(body, model_id, settings, *, source="openai-images",
                  input_images=None, cleanup_input_images=False):
    if input_images is not None and not 1 <= len(input_images) <= MAX_REFERENCE_IMAGES:
        raise ImageError(400, f"provide between 1 and {MAX_REFERENCE_IMAGES} reference images",
                         "image", "invalid_image_count")
    if not isinstance(body, dict):
        raise ImageError(400, "request must be a JSON object")
    fields = {"model", "prompt", "n", "size", "output_format", "output_compression",
              "response_format", "quality", "user", "seed", "steps"}
    for key in body.keys() - fields:
        raise ImageError(400, "unsupported request field", key, "unsupported_value")
    if body.get("model") != model_id:
        raise ImageError(400, "model must match the served model ID", "model")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
        raise ImageError(400, "prompt must contain 1–32000 characters", "prompt")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
        raise ImageError(400, "n must be 1 for this server", "n", "unsupported_value")
    for key, default, allowed in (
        ("quality", "auto", ("auto", "standard")),
        ("response_format", "b64_json", ("b64_json",)),
        ("output_format", "png", ("png", "jpeg", "webp")),
    ):
        if body.get(key, default) not in allowed:
            raise ImageError(400, f"unsupported {key}", key, "unsupported_value")
    fmt = body.get("output_format", "png")
    compression = body.get("output_compression")
    if "output_compression" in body and (
        type(compression) is not int or not 0 <= compression <= 100 or fmt == "png"
    ):
        raise ImageError(400, "output_compression must be 0–100 for JPEG/WebP", "output_compression")
    if "user" in body and not isinstance(body["user"], str):
        raise ImageError(400, "user must be a string", "user")
    size = body.get("size", "auto")
    if size == "auto":
        max_pixels = settings.get("max_image_pixels", 4194304)
        # Pick the largest model-supported square that fits the configured cap.
        side = (math.isqrt(max_pixels) // 16) * 16
        if side < 256:
            raise ImageError(400, "max_image_pixels must be at least 65536", "size")
        width = height = min(side, 1024)
    elif isinstance(size, str) and re.fullmatch(r"[0-9]{1,4}x[0-9]{1,4}", size):
        width, height = map(int, size.split("x"))
    else:
        raise ImageError(400, "size must be auto or WIDTHxHEIGHT", "size")
    if (any(d < 256 or d > 2048 or d % 16 for d in (width, height))
            or width * height > settings.get("max_image_pixels", 4194304)):
        raise ImageError(400, "size exceeds model/server limits (256–2048, multiples of 16)", "size")
    seed = body.get("seed", secrets.randbits(32))
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ImageError(400, "seed must be an integer in 0..4294967295", "seed")
    steps = body.get("steps")
    if steps is not None and (type(steps) is not int or not 1 <= steps <= 100):
        raise ImageError(400, "steps must be an integer in 1..100", "steps")
    request_type = ImageEditRequest if input_images is not None else ImageGenerationRequest
    extra = ({"input_images": input_images, "cleanup_input_images": cleanup_input_images}
             if input_images is not None else {})
    return request_type(prompt, width, height, fmt, compression, seed, steps,
                        source=source, **extra)


def stage_reference_image(data):
    """Validate, normalize and stage one upload for the private MLX worker."""
    import warnings
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not data or len(data) > MAX_REFERENCE_IMAGE_BYTES:
        raise ImageError(400, "reference image must be non-empty and at most 25 MiB",
                         "image", "invalid_image")
    path = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ("PNG", "JPEG", "WEBP"):
                    raise ImageError(400, "reference image must be PNG, JPEG, or WebP",
                                     "image", "unsupported_image_format")
                if source.width * source.height > MAX_REFERENCE_PIXELS:
                    raise ImageError(400, "reference image exceeds the 16-megapixel limit",
                                     "image", "image_too_large")
                source.load()
                normalized = ImageOps.exif_transpose(source).convert("RGB")
                try:
                    with tempfile.NamedTemporaryFile(prefix="mlxh-image-ref-", suffix=".png",
                                                     delete=False) as staged:
                        path = staged.name
                    normalized.save(path, format="PNG")
                finally:
                    normalized.close()
        return path
    except ImageError:
        if path:
            Path(path).unlink(missing_ok=True)
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        if path:
            Path(path).unlink(missing_ok=True)
        raise ImageError(400, "reference image exceeds safe decode limits",
                         "image", "image_too_large") from None
    except (OSError, ValueError, UnidentifiedImageError):
        if path:
            Path(path).unlink(missing_ok=True)
        raise ImageError(400, "invalid reference image", "image", "invalid_image") from None


def encode_image(image, request):
    class LimitedBuffer(io.BytesIO):
        def write(self, data):
            if self.tell() + len(data) > OUTPUT_LIMIT:
                raise ImageError(500, "encoded image exceeds 32 MiB", code="generation_failed")
            return super().write(data)

    if image.size != (request.width, request.height):
        raise ImageError(500, "backend returned incorrect dimensions", code="generation_failed")
    kwargs = {}
    if request.output_compression is not None:
        kwargs["quality"] = request.output_compression
    # Rebuild a clean image to strip backend EXIF/prompt metadata.
    from PIL import Image
    clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
    try:
        with LimitedBuffer() as buffer:
            clean.save(buffer, format=request.output_format.upper(), **kwargs)
            return buffer.getvalue()
    finally:
        clean.close()
