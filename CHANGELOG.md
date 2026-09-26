# Changelog

## Unreleased

## v0.2.0 — 2026-09-25

- **Model manager**: `mlxh serve` now starts a model-free API manager that loads
  independent model workers on demand, supports concurrent resident models,
  routes requests by model, and unloads idle workers. Status reports per-model
  worker state and activity. Fixed-model serving remains available with
  `mlxh serve NAME`.
- **Image generation and editing**: install optional image support, pull
  supported image models, and generate from the `mlxh image` interactive CLI
  or OpenAI-compatible Images API. The CLI supports seeds, steps, dimensions,
  output format/directory, local image references, and reference-based edits.
- **API instrumentation**: chat, image, and service requests use the shared
  engine/manager lifecycle and expose consistent request, token, timing,
  queue, and memory diagnostics.
- **Updates**: `mlxh update` / `mlxh upgrade` updates managed and uv-tool
  installations without touching models or configuration, and resynchronizes
  an installed image runtime. `mlxh --version` reports the package version.
- **Targeted logprobs**: non-streaming OpenAI chat completions can return
  full-vocabulary token log-probabilities, including the vLLM-compatible
  targeted-ID fields used by the frozen OpenJev helper.
- **Chat output**: streamed responses now use a live Rich Markdown renderer
  for headings, lists, tables, links, and fenced code blocks; pipes and
  `NO_COLOR` still receive plain output. The response footer includes total
  wall-clock execution time alongside output tokens and generation speed.
- **Chat input**: multiline terminal pastes remain one editable message and
  wait for an explicit Enter before generation starts.
- **Chat images**: Ctrl-V reads the macOS image clipboard and inserts an
  `[Image #N]` marker; `/image [path-or-URL]` also stages a clipboard, local,
  or bounded remote image. Pi registrations now advertise each local model's
  image-input capability, so Pi forwards images to vision models.

## v0.1.1 — 2026-09-23

- **Live diagnostics**: `GET /mlxh/info` now reports MLX memory, process
  uptime/readiness, queue activity, request and token counters, and a stable
  peak-memory snapshot. New `mlxh status` and `mlxh status --json` commands
  expose the same data for people and scripts.
- **Persistent server**: `mlxh service install|uninstall|restart` manages an
  opt-in per-user macOS LaunchAgent. It validates the configured model,
  forces localhost binding, safely handles occupied ports and reinstalls,
  writes an atomic plist, and integrates with `mlxh uninstall`.
- **Service reliability**: launcher discovery honors `MLXH_LAUNCHER`, load
  failures retain full tracebacks in `service.log`, and service lifecycle
  errors preserve the installed plist rather than leaving an unmanaged job.

## v0.1.0 — 2026-09-22

Alpha: developed and tested on one machine
(MacBook Pro M5 Pro, 48 GB, macOS 26); interfaces may change.

- **Models**: `search` (params/size/downloads from Hugging Face), `pull`
  (with provenance recorded as repo@revision), `link` (symlink existing
  dirs in), `list`, `mv`, `rm`, one models dir where the filesystem is
  the registry (`models_dir` config, `$MLXH_MODELS_DIR` override).
- **Chat**: interactive REPL with markdown-styled streaming, readline
  prompt history, image input (`/image`), `/reset`, `/exit`, `/help`,
  and an arrow-key model selector when no name is given. `run` pulls if
  needed and chats right away. No bundled tools: `--tools` (or the
  `chat_tools` config key) loads user-owned `~/.mlxh/tools.py`; a
  weather/time/calculator example ships in `examples/tools.py`.
- **Serve**: OpenAI-compatible API on port 1060 by default ("MLX" in
  Roman numerals — rare on purpose) (`/v1/chat/completions` streaming and
  non-streaming, `/v1/models`), tool calling (OpenAI `tools` in,
  `tool_calls` out), vision via `image_url` parts. Single dedicated
  generation thread (MLX streams are per-thread), disconnect-proof job
  queue, controls: `max_queued`, `max_tokens_cap`, `memory_limit_gb`,
  `cache_limit_gb`, `gen_timeout_s`.
- **Guardrails**: `pull` refuses models that don't fit free disk or
  unified memory (`--force` to override the memory check). The server
  auto-caps MLX memory at 80% of RAM and rejects prompts over
  `max_prompt_tokens` (default 8192) — a failed request instead of a
  frozen machine when a huge context balloons the KV cache.
- **Three API dialects**: OpenAI chat completions, the OpenAI Responses
  API (`/v1/responses`, what modern Codex requires), and the Anthropic
  Messages API.
- **Anthropic API + launch**: `serve` also speaks the Anthropic Messages
  API (`/v1/messages`, streaming and non-streaming, tool use, images,
  count_tokens), so Claude Code works against local models.
  `mlxh launch claude|codex|pi [--model NAME]` starts a server if
  needed, wires the agent (env vars, Codex model_provider overrides, or
  pi's models.json), and runs it (`--dry-run` prints the wiring;
  `--no-mcp` shrinks Claude Code's prompt for local models).
- **Home menu**: bare `mlxh` opens an interactive menu — launch a coding
  agent, chat, serve, or list models with arrow keys.
- **Prompt caching + thinking**: the server reuses KV blocks across
  requests (mlx-vlm automatic prefix caching; `prompt_cache` config) so
  same-prefix follow-ups skip prompt reprocessing. `thinking auto/on/off`
  controls reasoning models; templates that pre-open a think block have
  the reasoning stripped from API responses and rendered dimmed in chat
  (`--thinking` / `--no-thinking`).
- **Loaders**: stock MLX models via mlx-vlm/mlx-lm, Prism Hadamard packs
  via their bundled runtime, behind one Runner interface.
- **Install**: `curl | bash` (self-bootstrapping installer),
  `uv tool install git+…`, or `./install.sh` from a checkout; everything
  under `~/.mlxh`; `mlxh uninstall` removes it all.
