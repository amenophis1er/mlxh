# Local image generation

mlxh serves supported MFLUX image models through `POST /v1/images/generations`
and, for edit-capable models, `POST /v1/images/edits`. It runs one model per
server process; serve your language model on a different port if you need both
at once. FLUX.2 Klein 4B supports reference-image editing; Schnell and Qwen
Image currently support text-to-image only.

## Install and serve

```bash
mlxh images install
mlxh pull madroid/flux.1-schnell-mflux-4bit --kind image --name schnell
mlxh serve schnell --cache-limit-gb 1
```

Also supported by the pinned MFLUX 0.20.0 adapter:

```bash
mlxh pull black-forest-labs/FLUX.2-klein-4B --kind image --name klein
mlxh serve klein --image-steps 4

mlxh pull Qwen/Qwen-Image-2512 --kind image --name qwen-image
mlxh serve qwen-image --image-steps 20
```

The upstream FLUX.2 checkpoint is roughly 15 GB. Qwen Image has a much larger
memory and download footprint; check the pull size warning and available disk
and unified memory before proceeding. `--force` skips the RAM guard only.

`images install` installs MFLUX 0.20.0 and the calling mlxh code into a managed
environment under `MLXH_HOME/images`. It stages the installation before
activating it. Failed installations leave the previous environment intact;
logs are in `MLXH_HOME/images/install.log`. The active environment and one
previous version are retained for rollback; older staged environments are
pruned after a successful activation.

You do not need to install MFLUX yourself. Interactive image pulls offer to
install support if missing. Scripts must run `mlxh images install` explicitly.
Developers may instead use `uv run --extra images mlxh serve schnell`.

The standalone installer synchronizes an existing image environment on
upgrade. After an upgrade by another package manager, run `mlxh images install`
again. Startup refuses mismatched mlxh code, including stale editable builds.
Version and source checks happen before the server loads weights.

The FLUX.1 Schnell pack is pinned to revision
`4a5ef87e8f50a9d8576ea2f01e3bb4f00c5f1f5d`, downloads approximately 9.9 GB,
and includes its tokenizers and quantized text encoders. It loads locally
without another model download. Use a 32 GB or larger Apple Silicon machine
for the default 1024×1024 output; 256×256 smoke tests used about 12.1 GB of MLX
memory. Smaller-memory machines have not been qualified. Resolution, MLX
buffer caching, and other running models affect total memory requirements.
The model card declares Apache-2.0; review the model's license for your use.

## Edit with a reference image

Klein can use one or more images as references. This is image-conditioned
generation, not a pixel-preserving mask edit: details not mentioned in the
prompt may still change.

```bash
mlxh image klein --input-image ~/Pictures/butterfly.png \
  "Keep the shape and composition; add sharper hand-cut filigree to the lower wings" \
  --seed 42 --steps 4 --output ~/Pictures/butterfly-edited.png

mlxh image klein                    # interactive
# At the image> prompt:
# /ref ~/Pictures/butterfly.png
# Keep the shape and composition; add sharper filigree to the lower wings
```

Repeat `--input-image` or `/ref PATH` to provide up to four references.
Interactive references apply to the next submitted prompt and appear as
`[Image #N]` in the prompt label while pending. Press Tab after `/ref ` or
`/output ` to complete paths; `~` expands. `/clear-refs` clears pending
references. Without a pending reference, the prompt is text-to-image—not an
edit of the previous output. The API and CLI accept PNG, JPEG, and WebP; each input is
limited to 25 MiB and 16 megapixels, with a 50 MiB aggregate request limit.
The pinned Klein edit backend downsizes each reference aspect-preservingly to
at most 1 megapixel, then center-crops to dimensions divisible by 16. Fine
details may be lost during this preprocessing. Distilled Klein uses guidance
1.0; mlxh does not expose guidance, image strength, or KV-cache controls.
`/mlxh/info` advertises model edit support and effective limits.

## Generate with an OpenAI client

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1060/v1", api_key="local", timeout=600)
result = client.images.generate(
    model="schnell",
    prompt="A red canoe on a still alpine lake at sunrise",
    size="1024x1024",
    quality="standard",
    output_format="png",
    extra_body={"seed": 42},
)
Path("canoe.png").write_bytes(base64.b64decode(result.data[0].b64_json))
```

Use the exact name served by mlxh as `model`. The endpoint returns one image
in `data[0].b64_json`, its resolved `size` and `output_format`, and an `mlxh`
object with the seed, step count and elapsed generation time. A fixed seed is
repeatable with the same model, parameters, runtime and hardware; it is not a
cross-version reproducibility guarantee.

The CLI client uses the same local API and instrumented server engine:

```bash
mlxh image klein "A tiny red fox reading in a cozy bookstore" \
  --size 768x1024 --seed 42 --steps 4 \
  --output ~/Pictures/fox-bookstore.png
mlxh image klein --output-dir ~/Pictures   # interactive prompt loop
```

One-shot `--output` refuses to replace an existing file unless `--force` is
given. Without an exact output path, mlxh derives a filename from the prompt and
adds `-2`, `-3`, and so on to avoid collisions. In interactive mode, `/size`,
`/seed`, `/steps`, `/format`, and `/output` change options for later prompts.
Generated images default to the directory from which the command was started.

## Supported options

| Field | Supported values |
|---|---|
| `model` | Exact served model ID, required |
| `prompt` | Nonblank text, at most 32,000 characters and the selected model's tokenizer limit |
| `n` | Omitted or integer `1` |
| `size` | `auto` (1024×1024) or `WIDTHxHEIGHT`; 256–2048 per side, multiples of 16, within `max_image_pixels` |
| `output_format` | `png` (default), `jpeg`, `webp` |
| `output_compression` | JPEG/WebP encoder quality, integer 0–100; omitted uses Pillow defaults |
| `quality` | Omitted, `auto`, or `standard`: all use configured steps, response normalizes to `auto` |
| `response_format` | Omitted or `b64_json` |
| `seed` | mlxh extension, integer 0–4294967295; omitted picks and returns a random seed |
| `steps` | mlxh extension, optional integer 1–100; overrides the model/server default for this request |
| `user` | Optional string, ignored |

Prompt token limits are enforced before denoising using the selected model's
tokenizer. FLUX.1 Schnell's CLIP branch has a separate 77-token pooled-embedding
window and truncates there as part of normal FLUX behavior.

Unknown fields and unsupported options return 400, including `quality: hd` or
`high`, URL output, `moderation`, `background`, streaming and partial images.
Image edits support references only—no masks, inpainting, URL inputs, streaming,
or Responses API image tool. Local generation/editing does not reproduce
OpenAI moderation. Edit uploads are streamed through a 50 MiB body cap before
multipart parsing; generation JSON remains capped at 256 KiB.

## Operations

`mlxh status` shows IMAGES instead of token counts for an image server.
`mlxh status --json` includes current denoising progress, seed, dimensions,
memory, queue depth and the last outcome. Prompts are omitted from diagnostics
and generated files are never saved by the server. Backend prompt caches are
cleared after each request, and encoded image metadata excludes prompts.

The existing bounded queue and memory controls apply. Queue overflow returns
503. `gen_timeout_s` starts when the worker takes the job and returns 504 after
cooperative cancellation. Disconnects also cancel the job. Cancellation is
checked before MFLUX work and at denoising callbacks. Reference preprocessing
and VAE encoding happen before those callbacks and cannot be interrupted in
the middle; cancellation during preprocessing is observed when callbacks begin.
A queued cancelled job is skipped. Failed attempts do not increment IMAGES.

Request bodies are capped at 256 KiB before JSON parsing; encoded images are
capped at 32 MiB before base64. Images errors use the OpenAI error envelope.
`mlxh chat`, `run` and coding-agent launch reject image models. The existing
LaunchAgent can serve an image model through the normal service configuration.

References: [Images edit API](https://developers.openai.com/api/reference/cli/resources/images/methods/edit),
[MFLUX](https://github.com/mflux-community/mflux),
[verified model](https://huggingface.co/madroid/flux.1-schnell-mflux-4bit).
