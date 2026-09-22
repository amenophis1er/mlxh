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

```bash
git clone https://github.com/amenophis1er/mlxh.git
cd mlxh
./install.sh          # needs uv; installs to ~/.mlxh, launcher in ~/.local/bin
```

Override locations with `MLXH_HOME` / `MLXH_BIN`.

## Use

```bash
mlxh pull prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --name bonsai2   # download
mlxh link ~/some/existing/model --name mymodel                    # or register in place
mlxh list

mlxh chat bonsai2                        # interactive: /image <path>, /reset, Ctrl-D
mlxh chat bonsai2 -- -p "one question"   # one-shot (args after the name pass through)

mlxh serve bonsai2 --port 8081           # OpenAI-compatible API at /v1
mlxh config port 8082                    # persistent defaults

mlxh rm bonsai2                          # delete pulled weights (linked dirs: unregister only)
mlxh uninstall                           # remove ~/.mlxh + launcher, after confirmation
```

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
