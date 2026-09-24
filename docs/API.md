# mlxh serve — REST API reference

`mlxh serve <model>` exposes one model on `http://127.0.0.1:1060` (configurable
via `port`/`host`). Authentication: none — any bearer token / API key is
accepted. The server binds to localhost by default and must not be exposed
beyond it as-is.

Three client dialects are served side by side:

| Dialect | Endpoints | Typical clients |
|---|---|---|
| OpenAI chat | `POST /v1/chat/completions`, `GET /v1/models` | OpenAI SDKs, LangChain, most tooling |
| OpenAI Responses | `POST /v1/responses` | Codex (modern versions speak only this) |
| Anthropic | `POST /v1/messages`, `POST /v1/messages/count_tokens` | Anthropic SDKs, Claude Code |

Plus `GET /mlxh/info` (mlxh-specific).

## Execution model

One generation runs at a time (single GPU, no continuous batching). Further
requests queue, up to `max_queued`; beyond that the server answers **503**.
A job always runs to completion once started, even if the client disconnects.
With `prompt_cache` on (default), requests sharing a prefix with the previous
request skip reprocessing those tokens.

Server-side limits (see README "Server controls"):

- `max_tokens` is clamped to `max_tokens_cap` (default 16384).
- Prompts over `max_prompt_tokens` (default 8192) are rejected with **400**
  and a message naming the size and the config command to raise it.
- A generation exceeding `gen_timeout_s` (default 600) is stopped and
  returned as-is.
- MLX memory is capped (default 80% of RAM); allocations beyond it fail the
  request with **500**, not the machine.

Models whose chat template pre-opens a reasoning block (e.g. Bonsai) have the
`<think>…</think>` reasoning stripped from responses; the `thinking` setting
(`auto`/`on`/`off`) controls whether the model reasons at all.

## OpenAI dialect

### POST /v1/chat/completions

Request fields honored: `messages`, `tools`, `max_tokens` (or
`max_completion_tokens`), `temperature`, `top_p`, `stream`. The `model` field
is accepted and ignored — the server serves the model it was started with
(`GET /v1/models` tells you which).

Message content may be a string or OpenAI content parts. Image parts are
supported when the model has a vision tower:

```json
{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}
```

A local file path is also accepted as the `url` (localhost server, local
files).

Tool calling is symmetric with OpenAI: send `tools` (function specs), get
back `tool_calls` with `finish_reason: "tool_calls"`; return results as
`role: "tool"` messages. The server never executes tools. `tool_choice` is
not enforced (the model decides).

Streaming is standard SSE `chat.completion.chunk` events ending with
`data: [DONE]`; the final chunk carries `usage`. When `tools` are present the
output is buffered and delivered once parseable (tool-call XML must be read
whole). Comment lines (`: ping`) are emitted every 15s while the model is
still processing the prompt — SSE-legal, ignored by clients.

```bash
curl http://localhost:1060/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":100}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1060/v1", api_key="mlxh")
r = client.chat.completions.create(model="local", stream=True,
    messages=[{"role": "user", "content": "hi"}])
```

#### Token log-probabilities

Non-streaming chat completions can return full-vocabulary token
log-probabilities. This supports targeted-readout clients such as OpenJev.

| Field | Type | Behavior |
|---|---|---|
| `logprobs` | boolean | Include per-generated-token log-probabilities. |
| `top_logprobs` | integer, 0–128 | Include the top-k tokens from the full vocabulary. |
| `logprob_token_ids` | integer array | Also include these exact token IDs at every position. |
| `return_tokens_as_token_ids` | boolean | Return `token_id:N` labels instead of decoded strings. |
| `allowed_token_ids` | integer array | Accepted and validated for vLLM helper compatibility, but does not constrain sampling. |
| `chat_template_kwargs.enable_thinking` | boolean | Override the server's `thinking` setting for this request. |

`top_logprobs` is the descending union of the full-vocabulary top-k and all
requested `logprob_token_ids`, without renormalization. `allowed_token_ids`
is deliberately ignored because it affects vLLM sampling, while targeted
readout uses the unmasked scores. Normalization is computed in float32 even
when model logits are BF16. Streaming with `logprobs: true` returns **400**.

```json
{
  "choices": [{
    "message": {"role": "assistant", "content": "A"},
    "logprobs": {
      "content": [{
        "token": "A",
        "logprob": -0.0213,
        "bytes": [65],
        "top_logprobs": [
          {"token": "A", "logprob": -0.0213, "bytes": [65]},
          {"token": "C", "logprob": -4.11, "bytes": [67]}
        ]
      }]
    },
    "finish_reason": "stop"
  }]
}
```

Without `logprobs: true`, each choice contains `"logprobs": null`. With
`return_tokens_as_token_ids: true`, every token is formatted as
`token_id:N` and `bytes` is `null`. Otherwise, `bytes` is the UTF-8 encoding
of the decoded token; byte-fallback tokens may decode as U+FFFD, so this field
is best-effort. Log-probabilities describe raw generated tokens: tool-call
parsing and thinking suppression can rewrite the visible response afterward.
Use `chat_template_kwargs: {"enable_thinking": false}` when token/text
alignment matters.

### GET /v1/models

One entry: the loaded model, `id` = its mlxh name.

### POST /v1/responses

The OpenAI Responses API subset Codex needs. Honored: `instructions`,
`input` (string, or items: `message` with `input_text`/`input_image`/
`output_text` content, `function_call`, `function_call_output`), flat
`tools` (function type), `max_output_tokens`, `temperature`, `top_p`,
`stream`. Reasoning items are skipped.

Output items are `message` (with `output_text` content) and
`function_call`. Streaming follows the Responses event protocol:
`response.created`, `response.output_item.added`,
`response.content_part.added`, `response.output_text.delta` / `.done`,
`response.content_part.done`, `response.output_item.done`,
`response.completed` — with `: ping` comment keepalives during prompt
processing, and `response.failed` on errors.

## Anthropic dialect

### POST /v1/messages

Request fields honored: `system` (string or text blocks), `messages` with
content blocks (`text`, `image` base64 source, `tool_use`, `tool_result`),
`tools` (`name`/`description`/`input_schema`), `max_tokens`, `temperature`,
`top_p`, `stream`. System-role messages appearing mid-conversation (agent
"system reminders") are demoted to user messages rather than rejected.

Responses are Anthropic message objects: `content` blocks (`text`,
`tool_use`), `stop_reason` (`end_turn` / `max_tokens` / `tool_use`), `usage`
with `input_tokens`/`output_tokens`.

Streaming follows the Anthropic event protocol: `message_start` (emitted
immediately), `content_block_start` / `content_block_delta`
(`text_delta`, `input_json_delta`) / `content_block_stop`, `message_delta`
(with `stop_reason` and usage), `message_stop` — with `ping` events every
15s during prompt processing so agent clients don't mistake a slow model for
a dead network. Errors mid-stream arrive as an `error` event.

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:1060", api_key="mlxh")
r = client.messages.create(model="local", max_tokens=100,
    messages=[{"role": "user", "content": "hi"}])
```

### POST /v1/messages/count_tokens

Returns `{"input_tokens": N}` — a character-based estimate (length/4), good
enough for context budgeting, not exact.

## mlxh

### GET /mlxh/info

```bash
curl -s http://127.0.0.1:1060/mlxh/info | jq .
```

Returns the configuration and live diagnostics of this local server process:

```json
{
  "model": "bonsai2",
  "settings": {"max_queued": 4, "max_tokens_cap": 16384},
  "capabilities": {"images": true, "chat_protocol": 1},
  "mlx": {
    "active_memory_bytes": 13200000000,
    "cache_memory_bytes": 1200000000,
    "last_peak_memory_bytes": 13900000000
  },
  "runtime": {
    "engine_version": 1,
    "uptime_s": 3601,
    "pid": 12345,
    "ready": true,
    "busy": false,
    "queue_depth": 0,
    "requests": 42,
    "prompt_tokens": 98304,
    "tokens_generated": 183456,
    "mlx_version": "0.32.0",
    "current_request": null,
    "last_request": {
      "source": "chat",
      "duration_s": 4.8,
      "prompt_tokens": 230,
      "output_tokens": 91,
      "outcome": "completed"
    }
  }
}
```

Memory values are exact byte counts. `queue_depth` counts pending work, not the
active generation; `busy` reports that separately. `last_peak_memory_bytes` is
the stable peak snapshot from the last finished generation attempt (or model
load before the first request). Config edits after startup are not reflected
until restart. `mlxh status` presents this endpoint as a table and `mlxh status
--json` adds the locally discovered `port` to the payload.

`current_request` identifies live work and includes its source and elapsed
time; `last_request` is a single completion/failure/timeout/cancellation
snapshot, not retained history. The version and capability fields are used by
the terminal client to reject an old server cleanly.

`POST /mlxh/generate` and `DELETE /mlxh/requests/{id}` are versioned,
localhost-only implementation details used by `mlxh chat`; they are not public
compatibility endpoints. The private request body is capped at 36 MiB before
JSON parsing, and decoded images are capped at 25 MiB each.

## Not implemented

`n > 1`, streaming logprobs, `response_format`/JSON mode, enforced `tool_choice`,
`stop` sequences, embeddings, audio/video input, Anthropic `thinking` blocks
in responses (reasoning is stripped instead), and multi-model serving — one
server process serves one model; run several on different ports if needed.

## Image generation

An image-model server supports `POST /v1/images/generations` with the OpenAI
base64 response shape. Klein edit-capable servers also support
`POST /v1/images/edits` with multipart image references. See
[image generation and editing](IMAGES.md) for installation, supported fields,
SDK/CLI examples, limits, capabilities, and cancellation behavior.
