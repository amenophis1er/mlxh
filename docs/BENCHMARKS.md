# Benchmarks

Measured with mlxh's ancestor scripts on 2026-09-22. Hardware: MacBook Pro,
Apple M5 Pro, 48 GB unified memory, macOS 26.

## Ternary Bonsai 2 27B (MLX 2-bit vision pack)

Model: [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)
(8.6 GB on disk, vision tower included). Load time: ~3.3 s.

| Scenario | Prompt tokens | TTFT | Generation | Peak memory |
|---|---|---|---|---|
| Short prompt, 256 tokens out | 23 | 2.5 s (incl. warmup) | 22.0 tok/s | 9.3 GB |
| Short prompt, 1024 tokens out | 36 | 0.23 s | 22.2 tok/s | 9.3 GB |
| Long prompt (~8k tokens), 256 out | 8,083 | 25.3 s | 20.6 tok/s | 16.6 GB |

Notes:

- Generation speed is flat (~21–22 tok/s) regardless of output length.
- Prompt processing runs ~320 tok/s, so long-context requests pay real
  time-to-first-token; budget accordingly for RAG-style workloads.
- Memory grows with context: ~9.3 GB baseline, ~16.6 GB at 8k tokens.
- PrismML reports ~27.7 tok/s on M5 Pro for the text-only pack; the vision
  pack measured here is slightly slower.

## Reference point: stock small model

`mlx-community/Qwen2.5-0.5B-Instruct-4bit` (0.3 GB) through the same harness:
~118 tok/s generation on the same machine.
