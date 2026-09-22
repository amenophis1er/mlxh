# mlxh

A small harness for running local MLX models on Apple Silicon: pull weights
from Hugging Face, chat in the terminal (with tools and images), and serve an
OpenAI-compatible REST API. Everything it manages lives under one directory,
so uninstalling is complete and clean.

Supports two model families behind one interface:

- **Prism Hadamard packs** (e.g. `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`) —
  loaded through the runtime code bundled inside the pack.
- **Stock MLX models** (anything `mlx_vlm` / `mlx_lm` loads, e.g. most of
  `mlx-community/*`).

## Install

Quick install, no clone needed (requires [uv](https://docs.astral.sh/uv/)):

```bash
curl -fsSL https://raw.githubusercontent.com/amenophis1er/mlxh/main/install.sh | bash
```

Or from a checkout:

```bash
git clone https://github.com/amenophis1er/mlxh.git && cd mlxh && ./install.sh
```

Either way it installs to `~/.mlxh` with the launcher in `~/.local/bin`;
override locations with `MLXH_HOME` / `MLXH_BIN`. Re-running the installer
updates the app code in place (models and config are untouched).

## Use

```bash
mlxh pull prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --name bonsai2   # download
mlxh link ~/some/existing/model --name mymodel                    # or symlink one in
mlxh list

mlxh chat bonsai2                        # interactive: /image <path>, /reset, Ctrl-D
mlxh chat bonsai2 -- -p "one question"   # one-shot (args after the name pass through)

mlxh serve bonsai2 --port 8081           # OpenAI-compatible API at /v1
mlxh config port 8082                    # persistent defaults

mlxh rm bonsai2                          # delete pulled weights (links: symlink only)
mlxh uninstall                           # remove ~/.mlxh + launcher, after confirmation
```

All models live in one directory and the filesystem is the registry: any
subdirectory holding a `config.json` is usable — no bookkeeping. The location
is the `models_dir` config key (default `~/.mlxh/models`), overridable per
invocation with `$MLXH_MODELS_DIR`. `link` drops a symlink there, so linked
models show up like pulled ones but `rm` never touches the original files.

## Server controls

Set persistently with `mlxh config <key> <value>`, or per run with
`mlxh serve <name> --<key> <value>`:

| Key | Default | Meaning |
|---|---|---|
| `max_queued` | 4 | pending generations beyond the active one before 503 |
| `max_tokens_cap` | 16384 | server-side ceiling on requested `max_tokens` (0 = off) |
| `memory_limit_gb` | 0 (off) | MLX unified-memory limit for the server process |
| `cache_limit_gb` | 0 (off) | MLX buffer-cache limit (frees memory between requests) |
| `gen_timeout_s` | 600 | hard stop for a single generation (0 = off) |

Generation is intentionally serial (one at a time): Apple Silicon has one GPU
and the MLX stack has no continuous batching, so requests queue. For parallel
throughput, run a second `mlxh serve` instance on another port, or use a CUDA
box with vLLM / llama.cpp `--parallel`.

The API supports `/v1/chat/completions` (streaming + non-streaming),
`/v1/models`, tool calling (OpenAI `tools` in, `tool_calls` out), and vision
via `image_url` parts (base64 data URLs or local paths). Point any OpenAI
client at `http://localhost:<port>/v1` with any API key.

The chat CLI ships three built-in tools the model can call: live weather
(Open-Meteo), current time, and a safe calculator. Add your own in
`app/toolcalls.py` (`TOOL_REGISTRY` + `TOOL_SPECS`).

Throughput numbers from the machine this was built on are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Notes and limits

- One generation at a time (a second concurrent API request waits, then 503s).
- Tool-call parsing targets the Qwen-style `<tool_call>` XML format, which
  covers Qwen-lineage models (Bonsai included). Other families fall back to
  plain chat gracefully.
- Video input is not supported; images are.
- The server binds to 127.0.0.1 by default and has no authentication — don't
  expose it beyond localhost as-is.
- Prism packs execute loader code bundled with the downloaded weights. Review
  `runtime/*.py` in a pack before first use, the same way you'd review any
  `trust_remote_code` model.
