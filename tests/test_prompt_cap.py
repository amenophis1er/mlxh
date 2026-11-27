import json
from types import SimpleNamespace as NS

import pytest

from mlxh import prompt_cap, serve_app
from mlxh.engine import GenerationRequest, InferenceEngine
from mlxh.prompt_cap import PromptCap, cache_growth, context_length, memory_tokens, resolve

GB = 10**9


def model_dir(tmp_path, config, tokenizer=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if tokenizer is not None:
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    return tmp_path


@pytest.mark.parametrize("config,tokenizer,expected", [
    # the shapes of the models mlxh is tested with
    ({"max_position_embeddings": 32768}, {"model_max_length": 131072}, 32768),             # Qwen2.5
    ({"text_config": {"max_position_embeddings": 262144},
      "vision_config": {"hidden_size": 1152}}, {"model_max_length": 262144}, 262144),      # Bonsai
    ({"text_config": {"max_position_embeddings": 131072},
      "vision_config": {"max_position_embeddings": 4096}}, {"model_max_length": 10**30}, 131072),  # Gemma 4
    ({"text_config": {"model_max_length": 1048576},
      "mtp_config": {"max_position_embeddings": 512}}, {"model_max_length": 1048576}, 1048576),    # Inkling 2
    # the config says nothing: the tokenizer's limit, unless it is the "no limit" marker
    ({"hidden_size": 8}, {"model_max_length": 4096}, 4096),
    ({"hidden_size": 8}, {"model_max_length": 10**30}, None),
    ({"n_ctx": 2048, "n_positions": 1024}, None, 1024),
    ({"max_position_embeddings": True}, None, None),
    ({"max_position_embeddings": 0}, None, None),
])
def test_context_length_comes_from_the_text_model(tmp_path, config, tokenizer, expected):
    assert context_length(model_dir(tmp_path, config, tokenizer)) == expected


def test_context_length_of_a_missing_or_broken_model_is_unknown(tmp_path):
    assert context_length(tmp_path / "nowhere") is None
    (tmp_path / "config.json").write_text("{not json")
    assert context_length(tmp_path) is None
    (tmp_path / "config.json").write_text("[1, 2]")
    assert context_length(tmp_path) is None


def kv(slots, slot_bytes, window=None):
    """A key/value cache holding `slots` positions of `slot_bytes` each."""
    return NS(keys=NS(shape=(1, 8, slots, 128)), nbytes=slots * slot_bytes,
              **({"max_size": window} if window else {}))


def test_cache_growth_separates_what_grows_from_what_does_not():
    # Inkling 2 Mini after two tokens: 8 global layers, 40 windowed ones, 48 states
    cache = [NS(caches=(kv(256, 4096), NS(nbytes=39936))) for _ in range(8)]
    cache += [NS(caches=(kv(2, 4096, window=1024), NS(nbytes=39936))) for _ in range(40)]
    per_token, fixed = cache_growth(cache)
    assert per_token == 8 * 4096
    assert fixed == 40 * 1024 * 4096 + 48 * 39936


def test_cache_growth_of_an_unused_or_stateful_cache_is_all_fixed():
    per_token, fixed = cache_growth([NS(keys=None, nbytes=0), NS(nbytes=500),
                                     NS(keys=NS(shape=(1, 2)), nbytes=7)])
    assert (per_token, fixed) == (0, 507)


def test_memory_tokens_fits_the_copies_a_request_holds_into_what_is_free():
    cost = {"per_token": 32768, "fixed": 160 * 10**6, "copies": 2, "reserve": 12 * GB}
    free = (51 * GB - 16 * GB - 12 * GB) * prompt_cap.HEADROOM
    assert memory_tokens(51 * GB, 16 * GB, cost) == int((free - 2 * 160 * 10**6) / (2 * 32768))
    one_copy = memory_tokens(51 * GB, 16 * GB, {**cost, "copies": 1, "reserve": 0})
    assert one_copy > 2 * memory_tokens(51 * GB, 16 * GB, cost)


def test_memory_tokens_is_unknown_without_a_cost_or_a_limit_and_never_zero():
    cost = {"per_token": 1000, "fixed": 0}
    assert memory_tokens(0, 0, cost) is None
    assert memory_tokens(51 * GB, 0, None) is None
    assert memory_tokens(51 * GB, 0, {"per_token": 0, "fixed": 5}) is None
    assert memory_tokens(8 * GB, 9 * GB, cost) == prompt_cap.FLOOR  # the model alone is over the limit


@pytest.mark.parametrize("setting,context,memory,tokens,source", [
    (0, 1048576, 270000, 270000, "memory"),
    (0, 32768, 3000000, 32768, "context"),
    (0, 32768, None, 32768, "context"),
    (0, None, 50000, 50000, "memory"),
    (0, None, None, prompt_cap.FALLBACK, "fallback"),
    (40960, 1048576, 270000, 40960, "setting"),
    (500000, 32768, 1000, 500000, "setting"),
    (-1, 32768, 1000, None, "off"),
])
def test_resolve_takes_the_smaller_of_context_and_memory_unless_told(setting, context, memory, tokens, source):
    cap = resolve(setting, context, memory)
    assert (cap.tokens, cap.source) == (tokens, source)


def test_describe_names_the_reason():
    assert "what memory allows" in resolve(0, 1048576, 270000).describe()
    assert "1,048,576" in resolve(0, 1048576, 270000).describe()
    assert "its context length" in resolve(0, 32768, None).describe()
    assert "max_prompt_tokens" in resolve(40960).describe()
    assert "no limit" in resolve(-1).describe()


class Runner:
    supports_images = False

    def __init__(self, tokens):
        self.tokens = tokens

    def template(self, messages, **_kw):
        return list(range(self.tokens))

    def stream(self, *_a, **_kw):
        return iter(())


def engine_with(cap, setting=0, tokens=100):
    engine = InferenceEngine("/unused", "m", {**serve_app.SETTINGS, "max_prompt_tokens": setting})
    engine.runner, engine.prompt_cap = Runner(tokens), cap
    return engine


def run(engine):
    return list(engine._run_generation(GenerationRequest([{"role": "user", "content": "hi"}], "openai")))


def test_a_prompt_over_the_models_own_limit_is_refused_with_the_reason():
    engine = engine_with(PromptCap(50, "memory", 1048576, 50))
    with pytest.raises(Exception) as err:
        run(engine)
    assert err.value.status_code == 400
    assert "over this model's limit of 50 tokens, what memory allows" in err.value.detail
    assert "max_prompt_tokens 4196" in err.value.detail and "-1 = no limit" in err.value.detail


def test_a_prompt_at_the_limit_or_with_no_limit_goes_through():
    assert run(engine_with(PromptCap(100, "context", 100))) == []
    assert run(engine_with(PromptCap(None, "off"), tokens=10**6)) == []


def test_before_a_model_is_loaded_the_setting_alone_decides():
    engine = engine_with(None, setting=50)
    with pytest.raises(Exception, match="max_prompt_tokens"):
        run(engine)
    assert run(engine_with(None, setting=-1, tokens=10**5)) == []
    with pytest.raises(Exception, match="8,192"):
        run(engine_with(None, setting=0, tokens=9000))


def cap_at_load(monkeypatch, tmp_path, setting, cost, limit_gb=51.0, active=16 * GB):
    model_dir(tmp_path, {"text_config": {"model_max_length": 1048576}})
    engine = InferenceEngine(str(tmp_path), "m", {**serve_app.SETTINGS, "max_prompt_tokens": setting,
                                                    "memory_limit_gb": limit_gb})
    engine.runner = NS(prompt_cost=cost)
    monkeypatch.setattr("mlxh.cli.total_ram_bytes", lambda: 64 * GB)
    return engine._resolve_prompt_cap(NS(get_active_memory=lambda: active))


def test_at_load_the_cap_is_the_models_context_or_its_memory_estimate(monkeypatch, tmp_path):
    cost = {"per_token": 32768, "fixed": 0, "copies": 2, "reserve": 12 * GB}
    cap = cap_at_load(monkeypatch, tmp_path, 0, lambda: cost)
    assert cap.source == "memory" and cap.context_length == 1048576
    assert cap.tokens == memory_tokens(51 * GB, 16 * GB, cost)
    cheap = cap_at_load(monkeypatch, tmp_path, 0, lambda: {**cost, "per_token": 64})
    assert (cheap.tokens, cheap.source) == (1048576, "context")
    # with the MLX limit off, 80% of RAM stands in for it
    unlimited = cap_at_load(monkeypatch, tmp_path, 0, lambda: cost, limit_gb=-1)
    assert unlimited.tokens == memory_tokens(int(64 * GB * 0.8), 16 * GB, cost)


def test_a_cache_that_cannot_be_measured_leaves_the_context_length(monkeypatch, tmp_path, capsys):
    def broken():
        raise RuntimeError("no language_model")

    cap = cap_at_load(monkeypatch, tmp_path, 0, broken)
    assert (cap.tokens, cap.source) == (1048576, "context")
    assert "could not measure" in capsys.readouterr().out


def test_an_explicit_setting_is_used_as_is_without_measuring(monkeypatch, tmp_path):
    cap = cap_at_load(monkeypatch, tmp_path, 40960, lambda: pytest.fail("measured"))
    assert (cap.tokens, cap.source) == (40960, "setting")
    assert cap_at_load(monkeypatch, tmp_path, -1, lambda: pytest.fail("measured")).tokens is None


def test_info_reports_the_cap():
    engine = engine_with(PromptCap(270000, "memory", 1048576, 270000))
    assert engine.snapshot()["prompt_cap"] == {
        "tokens": 270000, "source": "memory", "context_length": 1048576, "memory_tokens": 270000}
    assert engine_with(None).snapshot()["prompt_cap"] is None
