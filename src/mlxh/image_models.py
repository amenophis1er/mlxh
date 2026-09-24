"""Image model catalog and metadata; independent of the optional runtime."""

import json
from pathlib import Path

MFLUX_VERSION = "0.20.0"

# Only models whose MFLUX adapter/configuration is exercised by mlxh are accepted.
# For raw upstream checkpoints the Hub SHA is resolved and recorded when pulled.
IMAGE_CATALOG = {
    "madroid/flux.1-schnell-mflux-4bit": {
        "backend": "mflux", "family": "flux1-schnell", "model_config": "schnell",
        "quantization_bits": 4, "supports_edits": False,
    },
    "black-forest-labs/FLUX.2-klein-4B": {
        "backend": "mflux", "family": "flux2-klein-4b", "model_config": "flux2_klein_4b",
        "quantization_bits": None, "supports_edits": True,
    },
    "Qwen/Qwen-Image-2512": {
        "backend": "mflux", "family": "qwen-image", "model_config": "qwen_image",
        "quantization_bits": None, "supports_edits": False,
    },
}

IMAGE_REPO = "madroid/flux.1-schnell-mflux-4bit"
IMAGE_REVISION = "4a5ef87e8f50a9d8576ea2f01e3bb4f00c5f1f5d"
IMAGE_METADATA = {"kind": "image", **IMAGE_CATALOG[IMAGE_REPO]}
FAMILY_ALIASES = {"flux-schnell": "flux1-schnell"}  # sidecars written by mlxh <= 0.1.x


def read_object(path):
    try:
        obj = json.loads(Path(path).read_text())
        return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def image_metadata(path):
    """Recognize a supported, explicitly tagged checkpoint or verified Schnell pack."""
    path = Path(path)
    meta = read_object(path / ".mlxh.json")
    if meta.get("kind") == "image":
        family = FAMILY_ALIASES.get(meta.get("family"), meta.get("family"))
        if meta.get("backend") != "mflux" or family not in {
            spec["family"] for spec in IMAGE_CATALOG.values()
        }:
            raise ValueError("unsupported image backend or family")
        spec = next(spec for spec in IMAGE_CATALOG.values() if spec["family"] == family)
        if meta.get("model_config", spec["model_config"]) != spec["model_config"]:
            raise ValueError("image model configuration does not match its family")
        bits = meta.get("quantization_bits")
        if bits != spec["quantization_bits"]:
            raise ValueError("image weights do not match declared quantization")
        return {"kind": "image", **spec}
    # Links to the verified downloaded snapshot can be identified without writes.
    config = read_object(path / "config.json")
    bits = (config.get("quantization_config") or {}).get("bits")
    card = path / "README.md"
    if bits in (4, 8) and card.is_file():
        if "base_model: black-forest-labs/FLUX.1-schnell" in card.read_text():
            if all((path / part).is_dir() for part in (
                "transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"
            )):
                return {**IMAGE_METADATA, "quantization_bits": bits}
    return None


def model_kind(path):
    return "image" if image_metadata(path) else "language"
