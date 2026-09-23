# mlxh

A small harness for running local MLX models on Apple Silicon: pull weights
from Hugging Face, chat in the terminal (with tools and images), and serve an
OpenAI-compatible REST API. Everything it manages lives under one directory,
so uninstalling is complete and clean.

**Status: alpha.** Built and tested on a single machine (MacBook Pro M5 Pro,
macOS 26). It works well there; expect rough edges elsewhere, and expect
interfaces to change. No support commitments.

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

Or as a uv tool (isolated env, managed by uv):

```bash
uv tool install git+https://github.com/amenophis1er/mlxh
```

Or from a checkout:

```bash
git clone https://github.com/amenophis1er/mlxh.git && cd mlxh && ./install.sh
```

Either way it installs to `~/.mlxh` with the launcher in `~/.local/bin`;
override locations with `MLXH_HOME` / `MLXH_BIN`. Re-running the installer
updates the app code in place (models and config are untouched).

## Use

Running bare `mlxh` opens an interactive home menu: launch a coding agent,
chat, serve, or list models — arrow keys and enter.

```bash
mlxh run mlx-community/Qwen2.5-0.5B-Instruct-4bit   # pull if needed, chat right away
mlxh search qwen3                        # find MLX models on Hugging Face
mlxh pull prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --name bonsai2   # download
mlxh link ~/some/existing/model --name mymodel                    # or symlink one in
mlxh list

mlxh chat bonsai2                        # interactive: /image <path>, /reset, Ctrl-D
mlxh chat bonsai2 -- --tools             # with your tools from ~/.mlxh/tools.py
mlxh chat bonsai2 -- -p "one question"   # one-shot (args after the name pass through)
mlxh chat bonsai2 -- --thinking          # show model reasoning, dimmed (--no-thinking skips it)

mlxh serve bonsai2                       # OpenAI + Anthropic API at :1060
mlxh launch claude --model gemma4-12b --no-mcp   # run Claude Code on a local model
mlxh config port 1234                    # persistent defaults

mlxh mv gemma-4-E4B-it-MLX-4bit gemma4   # rename to a nicer alias
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
| `memory_limit_gb` | 0 = auto | MLX memory limit; auto caps at 80% of RAM, -1 disables |
| `cache_limit_gb` | 0 (off) | MLX buffer-cache limit (frees memory between requests) |
| `gen_timeout_s` | 600 | hard stop for a single generation (0 = off) |
| `max_prompt_tokens` | 8192 | reject larger prompts with a 400 (0 = off) |
| `prompt_cache` | true | reuse KV blocks across requests — same-prefix follow-ups skip reprocessing (agents: 4-20x faster TTFT) |
| `thinking` | auto | model reasoning: `auto` (model default), `on`, `off` |

`mlxh status` shows live memory, queue, traffic, and process statistics for
the local server; use `mlxh status --json` for scripts.

To keep one model available across logins, opt in with
`mlxh config service_model bonsai2 && mlxh service install`. The per-user
LaunchAgent always binds to localhost and can be restarted with `mlxh service
restart`; remove it with `mlxh service uninstall` or `mlxh uninstall`.
LaunchAgents start at login rather than boot, hold the model in RAM while
running, and append unrotated output to `~/.mlxh/service.log`.

The last two exist because long contexts are a real hazard on unified memory:
KV cache grows with prompt length, and an unbounded 30k-token request can
exhaust RAM and freeze the whole machine. The server fails a request rather
than take the machine down; raise the limits deliberately when you have the
headroom (coding agents need `max_prompt_tokens` around 40960).

Generation is intentionally serial (one at a time): Apple Silicon has one GPU
and the MLX stack has no continuous batching, so requests queue. For parallel
throughput, run a second `mlxh serve` instance on another port, or use a CUDA
box with vLLM / llama.cpp `--parallel`.

The API supports `/v1/chat/completions` (streaming + non-streaming),
`/v1/models`, tool calling (OpenAI `tools` in, `tool_calls` out), and vision
via `image_url` parts (base64 data URLs or local paths). Point any OpenAI
client at `http://localhost:1060/v1` (the default port: MLX is the Roman
numeral for 1060) with any API key:

```bash
curl http://localhost:1060/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"bonsai2","messages":[{"role":"user","content":"hi"}],"max_tokens":100}'
```

The server also speaks the **Anthropic Messages API** (`/v1/messages`,
streaming with keepalive pings, tool use, images, `count_tokens`), which is
what Claude Code uses. `GET /mlxh/info` reports the live settings of a
running server. Full endpoint reference, request/response shapes, limits and
SDK examples: [docs/API.md](docs/API.md).

## Coding agents

```bash
mlxh launch claude --model gemma4-12b --no-mcp
mlxh launch codex --model gemma4-12b
mlxh launch pi --model gemma4-12b
mlxh launch claude --dry-run             # print the env + command instead
```

`launch` starts a server if none is running (and stops it again when the
agent exits), wires the agent up, and runs it. Anything after `--` passes
through to the agent. claude is wired via `ANTHROPIC_BASE_URL`; codex via
`-c model_provider` overrides against the server's Responses API (works with
ChatGPT-account Codex, which ignores `OPENAI_BASE_URL`); pi by registering an
`mlxh` provider in `~/.pi/agent/models.json` (merged non-destructively, with
the compat flags plain OpenAI-compatible servers need).

Practical notes:

- Coding agents send very large prompts. `--no-mcp` (claude only) launches
  without MCP servers, cutting the prompt from ~50k to ~16k tokens — start
  there. For full-MCP sessions, raise `max_prompt_tokens` accordingly and
  make sure the model + context fits your memory.
- The prompt cache makes follow-up turns fast; the first turn still pays
  full prompt processing (~40s on a 12B for Claude Code's core prompt).
- If the local model errors, Claude Code may silently fall back to a real
  Anthropic model on your account — watch for its "Switched to …" notice.
- Model advice: a 12B-class stock model works well; heavily-compressed
  large models (Bonsai) are slow at agent-scale contexts.

## Chat images

Vision models accept local files, web images, and the macOS image clipboard:

```text
/image ~/Desktop/screenshot.png
/image https://example.com/diagram.png
/image
```

The no-argument form reads the current clipboard image. Terminals do not pass
binary clipboard contents through ordinary Cmd-V, so this explicit command is
used instead. URL downloads are limited to 25 MB, validated as images, and—like
clipboard captures—removed after the next message is processed.

## Chat tools

mlxh ships **no tools** — by default the chat is a plain model REPL. To give
the model tools, create `~/.mlxh/tools.py` (a file you own) and start a chat
with `--tools`, or make that the default with `mlxh config chat_tools on`.
The contract is two module-level names:

- `TOOL_REGISTRY`: `{"tool_name": callable(**kwargs) -> dict}`
- `TOOL_SPECS`: the same tools described in the OpenAI function-tool format

A ready-made example (live weather via Open-Meteo, current time, a safe
calculator) ships in [examples/tools.py](examples/tools.py):

```bash
cp examples/tools.py ~/.mlxh/tools.py
mlxh chat <model> -- --tools
```

Every tool execution is printed as it happens, and tools run with your user's
permissions — only put code in `tools.py` that you'd run yourself. The API
server never executes tools: it returns `tool_calls` to the API client in the
standard OpenAI format, and the client runs its own.

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
