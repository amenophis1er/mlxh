# Changelog

## v0.1.0 — unreleased

Alpha: developed and tested on one machine
(MacBook Pro M5 Pro, 48 GB, macOS 26); interfaces may change.

- **Models**: `search` (params/size/downloads from Hugging Face), `pull`
  (with provenance recorded as repo@revision), `link` (symlink existing
  dirs in), `list`, `mv`, `rm`, one models dir where the filesystem is
  the registry (`models_dir` config, `$MLXH_MODELS_DIR` override).
- **Chat**: interactive REPL with streaming, image input (`/image`),
  history (`/reset`), built-in tools (opt-in via `--tools` or the
  `chat_tools` config key: live weather, time, calculator), and an
  arrow-key model selector when no name is given. `run` pulls if needed
  and chats right away.
- **Serve**: OpenAI-compatible API (`/v1/chat/completions` streaming and
  non-streaming, `/v1/models`), tool calling (OpenAI `tools` in,
  `tool_calls` out), vision via `image_url` parts. Single dedicated
  generation thread (MLX streams are per-thread), disconnect-proof job
  queue, controls: `max_queued`, `max_tokens_cap`, `memory_limit_gb`,
  `cache_limit_gb`, `gen_timeout_s`.
- **Guardrails**: `pull` refuses models that don't fit free disk or
  unified memory (`--force` to override the memory check).
- **Loaders**: stock MLX models via mlx-vlm/mlx-lm, Prism Hadamard packs
  via their bundled runtime, behind one Runner interface.
- **Install**: `curl | bash` (self-bootstrapping installer),
  `uv tool install git+…`, or `./install.sh` from a checkout; everything
  under `~/.mlxh`; `mlxh uninstall` removes it all.
