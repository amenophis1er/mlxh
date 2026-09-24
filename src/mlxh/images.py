"""Typed image requests and strict OpenAI Images protocol validation."""

import base64
import io
import math
import re
import secrets
from dataclasses import dataclass

from .engine import EngineError

BODY_LIMIT = 256 * 1024
OUTPUT_LIMIT = 32 * 1024 * 1024


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
class ImageGenerationResult:
    image: bytes
    request: ImageGenerationRequest
    steps: int
    generation_s: float

    def response(self, created):
        r = self.request
        return {"created": created, "data": [{"b64_json": base64.b64encode(self.image).decode()}],
                "output_format": r.output_format, "size": f"{r.width}x{r.height}",
                "quality": "auto", "mlxh": {"seed": r.seed, "steps": self.steps,
                                             "generation_s": round(self.generation_s, 3)}}


def parse_request(body, model_id, settings):
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
    return ImageGenerationRequest(prompt, width, height, fmt, compression, seed, steps)


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
