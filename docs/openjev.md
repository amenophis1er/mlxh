# Running OpenJev through mlxh

OpenJev-style decision models render lettered choices and classify by reading
the first output token's log-probabilities. mlxh supports the frozen OpenJev
helper through non-streaming `/v1/chat/completions`, including its targeted
token-ID fields.

## Establish a direct MLX baseline

The helper scripts live in the main `openjev/openjev` repository, not in its
published MLX weight repositories. Download the two small files separately;
the frozen `shim.py` used for the published results has SHA `81a22f1b…`:

```bash
mlxh pull openjev/openjev-MLX-4bit
mkdir -p ~/.mlxh/openjev-helper
OPENJEV_REV=5ec9e5fd2f80a6fff386779b1e5ac7e389971889
curl -fL https://huggingface.co/openjev/openjev/resolve/$OPENJEV_REV/helper/shim.py \
  -o ~/.mlxh/openjev-helper/shim.py
curl -fL https://huggingface.co/openjev/openjev/resolve/$OPENJEV_REV/helper/shim_mlx.py \
  -o ~/.mlxh/openjev-helper/shim_mlx.py
cd ~/.mlxh/openjev-helper
printf '%s  %s\n' 81a22f1b1b8912a465059207ef9f60b7c6c16b4de6372305d867efbe38a1987a shim.py \
  | shasum -a 256 -c -
READOUT_T=0.85 READOUT_NOUL_T=1.829074 READOUT_NOUL_BIAS=0 READOUT_TARGETED=1 \
READOUT_INSTR_STYLE=pyrepr SHIM_STAGGER=1 \
uv run --python 3.11 --with 'mlx>=0.32,<0.33' --with 'mlx-lm==0.31.3' \
  --with 'transformers>=5.5' python shim_mlx.py --helper shim.py \
  --model ~/.mlxh/models/openjev-MLX-4bit --selfcheck
```

The published `openjev/openjev-MLX` build is 8-bit and about 27 GB;
`openjev/openjev-MLX-4bit` is about 15 GB and slightly less accurate. Both are
text-only. Keep the self-check letter log-probabilities as the fidelity target
for the server path.

## Run the frozen helper against mlxh

```bash
mlxh serve bonsai2

VLLM=http://127.0.0.1:1060/v1 \
TOKENIZER=~/.mlxh/models/bonsai2 \
READOUT_T=0.85 READOUT_NOUL_T=1.829074 READOUT_NOUL_BIAS=0 \
READOUT_TARGETED=1 READOUT_INSTR_STYLE=pyrepr SHIM_PAD=16 \
uv run --python 3.11 --with openai --with 'transformers>=5.5' \
  python ~/.mlxh/openjev-helper/shim.py --port 3000
```

You can substitute `openjev-MLX-4bit` or `openjev-MLX` for `bonsai2`. The
model-card request works unchanged against `localhost:3000/v1/systemone`.
The helper's hardcoded `model: "qwen"` value is harmless in the fixed-model
mode shown above: mlxh serves `bonsai2` and ignores the request's `model`
field. If using manager mode (`mlxh serve` without a model), change that field
to the installed model name (`bonsai2`) so the manager routes correctly.

Keep `READOUT_TARGETED=1`: it requests every letter by token ID, avoiding a
missing letter outside top-k. `SHIM_PAD=16` aligns repeated VLM prompts with
mlxh's APC cache block size and is irrelevant for text-only runners.

The calibration constants above were fitted on FP8 OpenJev under vLLM and
transfer to the published MLX 8-bit build according to its model card.
`bonsai2` is ternary and not decision-tuned, so treat them as starting values
and refit on a small labeled sample from your domain. Apple Silicon latency is
prompt-bound—expect seconds for a fresh 1k-token state on a 27B model, not the
model card's GPU latency.

For fidelity testing, send the same question to the direct `shim_mlx.py`
server and the mlxh-backed helper. With the same model and prompt, calibrated
probabilities should agree to roughly three decimal places. A larger mismatch
usually means the chat template or thinking mode differs; the helper sends
`chat_template_kwargs.enable_thinking=false` to avoid that ambiguity.
