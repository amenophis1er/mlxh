# Changelog

## Unreleased

- **Prompt cache across agent turns**: a cache that cannot be trimmed
  (Inkling 2, Qwen 3.5-style hybrids loaded through mlx-lm) is also kept as
  it stood at the end of each prompt, so the next turn reuses everything up
  to the end of the previous prompt even though clients send the history
  back without the model's reasoning. A conversation keeps its two latest
  snapshots and its latest reply; older ones and the replies its history
  dropped are removed, so the cap is not charged again for the same turns,
  and a side request that forks the history does not cost the next turn its
  cache. `prompt_cache_gb` counts stored entries; a request in flight holds
  up to two more copies of its own context. Trimmable caches are unchanged.
- **Cached tokens in usage**: `prompt_tokens_details.cached_tokens` on chat
  completions, `input_tokens_details.cached_tokens` on Responses, and
  `cache_read_input_tokens` on Anthropic messages (whose `input_tokens` now
  leaves them out, as Anthropic's does; `cache_creation_input_tokens` is
  always 0). `mlxh status` shows each resident worker's prompt cache.
- **Reasoning effort per request**: chat completions `reasoning_effort`,
  Responses `reasoning.effort` (`none` … `high`, plus `xhigh`/`max`) and
  Anthropic `thinking` (`disabled`, or `enabled` with `budget_tokens`: under
  4096 is `low`, under 16384 `medium`, else `high`). Inkling 2 gets the level
  in its template; other models switch thinking off for `none` and on
  otherwise. The request wins over `enable_thinking` and the `thinking`
  setting; unknown values are a 400. Behaviour change: agent clients that
  send these fields unasked (Claude Code's `thinking`, Codex's
  `reasoning.effort`) now override `thinking off`/`on` on the server, where
  the fields were ignored before.
- **Reasoning in API responses**: the model's reasoning, which was dropped,
  now comes back beside the answer: `reasoning_content` on chat completions
  (message and stream deltas), a signed `thinking` block on Anthropic, and a
  `reasoning` output item on Responses. Answer text stays free of reasoning,
  and streamed text still stops at a tool call.
- Stream fixes that came with it. Responses streaming held back the last 11
  characters of every answer (all of a short one such as `Paris`) and never
  sent them as deltas; streamed text now waits only for a tail that could
  still become a tool call. Its `output_item.done` ids match the ones
  streamed. On Anthropic and Responses streams, a tool call that does not
  parse and any text after a tool call are now sent as text, as the
  non-streaming responses have them. `mlxh chat` no longer drops characters
  from an answer that starts with whitespace.

## v0.2.2 — 2026-11-25

- **Inkling 2 Mini** (`mlx-community/Inkling-2-Mini-4bit`): served from its own
  runtime, `mlxh inkling install` (offered on `mlxh pull`), with mlx-lm pinned
  to the reviewed commit of ml-explore/mlx-lm#2086. mlxh's own environment stays
  on mlx-lm 0.31.3, because newer mlx-lm moves the classes the Prism (Bonsai)
  packs import. Inkling's tagged messages map onto mlxh's reasoning split and
  tool calls; thinking `off` asks for no effort. Text only for now; the
  `-oQ4e-mtp` packing does not load with that commit. Its prompt cache cannot
  be trimmed, so only a strict prefix is reused across turns.
- The image runtime's installer is now shared by both optional runtimes;
  `mlxh update` refreshes each one present.

## v0.2.1 — 2026-11-11

- **Prefix cache for text-only models**: `TextRunner` (models loaded through
  mlx-lm, including Prism text packs) now reuses the KV cache across requests
  via mlx-lm's `LRUPromptCache`, so agent follow-up turns only process new
  tokens. Bounded by the new `prompt_cache_gb` setting (0 = auto: 25% of the
  memory limit) and 32 entries; `mlxh status` shows entries, size and hits.
  Each response carries `cached_tokens` for the usage-reporting work in #5.
- **Image runtime**: pin Pillow below 13.0, which fails to import on
  Python 3.11 and broke `mlxh images install`; failed staging environments
  (~1.3 GB each) are now removed instead of accumulating.
- **Install/service fixes**: `install.sh` works under macOS bash 3.2 without
  `GITHUB_TOKEN`; `mlxh service restart` waits for launchd to drop the job;
  the manager refuses unknown paths before loading a worker and rejects
  non-numeric `--*-gb` flags up front.
- **CI**: the Apple Silicon job runs on macos-15 (macos-14 was retired).

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
