"""TextRunner prefix cache: reuse, exact-hit step-back, bounds, and stats."""
import copy

import pytest

mlx_lm = pytest.importorskip("mlx_lm", reason="prefix cache tests need mlx-lm (Apple Silicon)")
cache_mod = pytest.importorskip("mlx_lm.models.cache")

from mlxh import loader
from mlxh.loader import TextRunner


class FakeKV:
    """Stands in for a KVCache: trimmable, tracks how many tokens it holds."""

    def __init__(self):
        self.offset = 0

    @property
    def nbytes(self):
        return self.offset * 1000

    @property
    def state(self):
        return self.offset

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n


class FakeLinear(FakeKV):
    def is_trimmable(self):
        return False


class Model:
    """Hashable stand-in; the trie keys entries by model. A forward pass (the
    snapshot prefill) adds its tokens to every layer, as stream_generate does."""

    def __init__(self, trimmable=True):
        self.trimmable = trimmable
        self.calls = []

    def __call__(self, inputs, cache):
        self.calls.append(inputs.tolist()[0])
        for layer in cache:
            layer.offset += inputs.shape[1]


class Tokenizer:
    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


class Resp:
    def __init__(self, token, prompt_tokens, finish_reason=None):
        self.token = token
        self.text = chr(97 + token % 26)
        self.prompt_tokens = prompt_tokens
        self.prompt_tps = 100.0
        self.generation_tokens = 1
        self.finish_reason = finish_reason


def fake_stream_generate(model, tokenizer, prompt, max_tokens=1, **kwargs):
    """Append len(prompt) to each layer cache, then emit `max_tokens` tokens."""
    fake_stream_generate.calls.append(list(prompt))
    prompt_cache = kwargs.get("prompt_cache")
    assert prompt_cache is not None, "runner must always pass a cache"
    assert len(prompt) > 0, "mlx_lm raises on an empty prompt"
    for layer in prompt_cache:
        layer.offset += len(prompt)
    for i in range(max_tokens):
        for layer in prompt_cache:
            layer.offset += 1
        yield Resp(100 + i, len(prompt), "stop" if i == max_tokens - 1 else None)


@pytest.fixture
def runner(monkeypatch):
    fake_stream_generate.calls = []
    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    monkeypatch.setattr(
        cache_mod, "make_prompt_cache",
        lambda model: [FakeLinear() if not model.trimmable else FakeKV(), FakeKV()],
    )
    r = TextRunner(Model(), Tokenizer())
    r.enable_cache(max_bytes=10**9, max_entries=4)
    return r


def run(runner, prompt, max_tokens=2):
    return list(runner.stream(prompt, max_tokens=max_tokens))


def test_first_request_processes_whole_prompt_and_stores_it(runner):
    out = run(runner, [1, 2, 3, 4])
    assert fake_stream_generate.calls == [[1, 2, 3, 4]]
    assert [r.cached_tokens for r in out] == [0, 0]
    assert runner.cache_stats()["entries"] == 1
    assert runner.cache_stats()["misses"] == 1


def test_follow_up_turn_only_processes_new_tokens(runner):
    run(runner, [1, 2, 3, 4])                     # stored key: [1,2,3,4,100,101]
    out = run(runner, [1, 2, 3, 4, 100, 101, 5, 6])  # agent echoes the reply, adds a turn
    assert fake_stream_generate.calls[-1] == [5, 6]
    assert out[0].cached_tokens == 6
    assert out[0].prompt_tokens == 8, "usage must report the full prompt"
    stats = runner.cache_stats()
    assert stats["hits"] == 1 and stats["cached_tokens_total"] == 6


def test_exact_hit_steps_back_one_token_instead_of_empty_prompt(runner):
    run(runner, [1, 2, 3, 4])
    run(runner, [1, 2, 3, 4, 100, 101])  # identical to a stored key
    assert fake_stream_generate.calls[-1] == [101]


def test_shorter_prompt_trims_a_longer_cached_entry(runner):
    run(runner, [1, 2, 3, 4, 5, 6, 7, 8])
    run(runner, [1, 2, 3, 4, 9])  # shares a 4-token prefix with the stored entry
    assert fake_stream_generate.calls[-1] == [9]


def test_one_token_prompt_against_longer_entry_is_a_plain_miss(runner):
    run(runner, [1, 2, 3, 4])
    out = run(runner, [1])  # nothing can stay cached: discard the copy
    assert fake_stream_generate.calls[-1] == [1]
    assert out[0].cached_tokens == 0
    assert runner.cache_stats()["misses"] == 2


@pytest.fixture
def linear(monkeypatch):
    """A runner whose cache cannot be trimmed (Inkling, Qwen 3.5 hybrids)."""
    fake_stream_generate.calls = []
    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    monkeypatch.setattr(cache_mod, "make_prompt_cache",
                        lambda model: [FakeLinear(), FakeKV()])
    r = TextRunner(Model(trimmable=False), Tokenizer())
    r.enable_cache(max_bytes=10**9)
    return r


def stored_keys(runner):
    return sorted(tokens for queue in runner.prompt_cache._lru._lrus.values()
                  for _, tokens in queue)


def test_non_trimmable_cache_keeps_a_snapshot_at_the_end_of_the_prompt(linear):
    run(linear, [1, 2, 3, 4])
    # The model prefilled all but the last token, mlx-lm processed that one.
    assert linear.model.calls == [[1, 2, 3]]
    assert fake_stream_generate.calls == [[4]]
    assert stored_keys(linear) == [[1, 2, 3], [1, 2, 3, 4, 100, 101]]


def test_non_trimmable_next_turn_reuses_the_prompt_when_the_reply_is_rerendered(linear):
    run(linear, [1, 2, 3, 4])                  # replies 100, 101 (with reasoning)
    # The history comes back without the reasoning: 50, 51 instead of 100, 101.
    out = run(linear, [1, 2, 3, 4, 50, 51, 5, 6])
    assert out[0].cached_tokens == 3
    assert linear.model.calls[-1] == [4, 50, 51, 5]
    assert fake_stream_generate.calls[-1] == [6]
    out = run(linear, [1, 2, 3, 4, 50, 51, 5, 6, 52, 7, 8])
    assert out[0].cached_tokens == 7
    assert fake_stream_generate.calls[-1] == [8]
    # Earlier snapshots and the replies the history dropped are not kept.
    assert stored_keys(linear) == [[1, 2, 3, 4, 50, 51, 5, 6, 52, 7],
                                   [1, 2, 3, 4, 50, 51, 5, 6, 52, 7, 8, 100, 101]]
    stats = linear.cache_stats()
    assert stats["entries"] == 2
    assert stats["bytes"] == 2 * 1000 * (10 + 13)


def test_non_trimmable_reply_echoed_exactly_is_still_reused(linear):
    run(linear, [1, 2, 3, 4])
    out = run(linear, [1, 2, 3, 4, 100, 101, 7])
    assert out[0].cached_tokens == 6
    assert fake_stream_generate.calls[-1] == [7]


def test_non_trimmable_divergent_or_repeated_prompt_reuses_the_snapshot(linear):
    run(linear, [1, 2, 3, 4])
    run(linear, [1, 2, 3, 9])                 # differs only in the last token
    assert fake_stream_generate.calls[-1] == [9]
    assert linear.model.calls[-1] == [1, 2, 3]
    run(linear, [1, 2, 3, 9])                 # exact repeat: one token, no new snapshot
    assert fake_stream_generate.calls[-1] == [9]
    assert len(linear.model.calls) == 1


def test_non_trimmable_other_conversations_are_not_dropped(linear):
    run(linear, [1, 2, 3, 4])
    run(linear, [7, 8, 9])
    run(linear, [1, 2, 3, 4, 50, 5])
    assert [7, 8] in stored_keys(linear)
    assert [7, 8, 9, 100, 101] in stored_keys(linear)
    out = run(linear, [7, 8, 9, 50, 6])
    assert out[0].cached_tokens == 2


def test_non_trimmable_failed_or_unsound_requests_store_no_snapshot(linear, monkeypatch):
    def exploding(model, tokenizer, prompt, max_tokens=1, **kwargs):
        yield Resp(100, len(prompt))
        raise RuntimeError("metal oom")

    monkeypatch.setattr(mlx_lm, "stream_generate", exploding)
    with pytest.raises(RuntimeError):
        run(linear, [1, 2, 3])
    assert linear.cache_stats()["entries"] == 0

    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    monkeypatch.setattr(TextRunner, "_cache_is_sound", staticmethod(lambda cache: False))
    gen = linear.stream([1, 2, 3], max_tokens=5)
    next(gen)
    gen.close()
    assert linear.cache_stats()["entries"] == 0


def test_non_trimmable_failed_prefill_keeps_earlier_entries(linear, monkeypatch):
    run(linear, [1, 2, 3, 4])

    def failing(self, cache, tokens, step=2048):
        raise RuntimeError("metal oom")

    monkeypatch.setattr(TextRunner, "_prefill_snapshot", failing)
    with pytest.raises(RuntimeError):
        run(linear, [1, 2, 3, 4, 50, 5])
    assert stored_keys(linear) == [[1, 2, 3], [1, 2, 3, 4, 100, 101]]


def test_non_trimmable_cancel_with_sound_cache_keeps_the_snapshot(linear):
    gen = linear.stream([1, 2, 3], max_tokens=5)
    next(gen)
    gen.close()
    assert [1, 2] in stored_keys(linear)


def test_snapshot_prefill_uses_generate_steps_chunks(linear):
    cache = cache_mod.make_prompt_cache(linear.model)
    snapshot = linear._prefill_snapshot(cache, [1, 2, 3, 4, 5], step=2)
    assert linear.model.calls == [[1, 2], [3, 4], [5]]
    assert [layer.offset for layer in snapshot] == [5, 5]
    cache[0].offset += 1  # generation goes on in the live cache...
    assert snapshot[0].offset == 5  # ...not in the copy


def test_snapshot_reports_prompt_speed_over_every_processed_token(linear, monkeypatch):
    times = iter([0.0, 2.0])
    monkeypatch.setattr("mlxh.loader.time.perf_counter", lambda: next(times))
    out = run(linear, [1, 2, 3, 4, 5])  # 4 tokens prefilled in 2 s, 1 by mlx-lm at 100/s
    assert out[0].prompt_tps == pytest.approx(5 / 2.01)


def test_trimmable_cache_takes_no_snapshot(runner):
    run(runner, [1, 2, 3, 4])
    assert runner.model.calls == []
    assert fake_stream_generate.calls == [[1, 2, 3, 4]]


def test_cache_is_bounded_by_bytes_and_entries(runner):
    runner.enable_cache(max_bytes=7 * 1000 * 2, max_entries=2)  # 2 layers * 7 tokens
    run(runner, [1, 2, 3, 4])         # 6 tokens stored
    run(runner, [11, 12, 13, 14])     # second entry; 6 + 6 > cap -> evict LRU
    stats = runner.cache_stats()
    assert stats["entries"] == 1
    assert stats["bytes"] <= stats["max_bytes"]


def test_cancelled_generation_still_stores_generated_prefix(runner):
    gen = runner.stream([1, 2, 3], max_tokens=5)
    next(gen)
    gen.close()
    assert runner.cache_stats()["entries"] == 1
    run(runner, [1, 2, 3, 100, 7])
    assert fake_stream_generate.calls[-1] == [7]


def test_failed_generation_does_not_store_a_half_updated_cache(runner, monkeypatch):
    def exploding(model, tokenizer, prompt, max_tokens=1, **kwargs):
        for layer in kwargs["prompt_cache"]:
            layer.offset += len(prompt) + 1
        yield Resp(100, len(prompt))
        raise RuntimeError("metal oom")

    monkeypatch.setattr(mlx_lm, "stream_generate", exploding)
    with pytest.raises(RuntimeError):
        run(runner, [1, 2, 3])
    assert runner.cache_stats()["entries"] == 0


def test_hit_reports_full_prompt_but_measured_tps(runner):
    run(runner, [1, 2, 3, 4])
    out = run(runner, [1, 2, 3, 4, 100, 101, 5, 6])
    assert out[0].prompt_tokens == 8 and out[0].cached_tokens == 6
    assert out[0].prompt_tps == 100.0


def test_failed_hit_evicts_nothing(runner, monkeypatch):
    run(runner, [1, 2, 3, 4])
    run(runner, [11, 12, 13, 14])
    before = runner.cache_stats()["entries"]

    def exploding(model, tokenizer, prompt, max_tokens=1, **kwargs):
        raise RuntimeError("metal oom")
        yield  # noqa: unreachable, makes this a generator

    monkeypatch.setattr(mlx_lm, "stream_generate", exploding)
    with pytest.raises(RuntimeError):
        run(runner, [1, 2, 3, 4, 100, 7])
    assert runner.cache_stats()["entries"] == before


def test_stored_bytes_stay_under_the_cap_after_a_hit(runner):
    runner.enable_cache(max_bytes=8 * 1000 * 2, max_entries=4)
    run(runner, [1, 2, 3, 4])          # 6 tokens stored
    run(runner, [1, 2, 3, 4, 100, 7])  # superseding entry replaces its prefix
    stats = runner.cache_stats()
    assert stats["bytes"] <= stats["max_bytes"]
    assert stats["entries"] == 1


def test_cancel_with_unsound_cache_stores_nothing(runner, monkeypatch):
    monkeypatch.setattr(TextRunner, "_cache_is_sound", staticmethod(lambda cache: False))
    gen = runner.stream([1, 2, 3], max_tokens=5)
    next(gen)
    gen.close()
    assert runner.cache_stats()["entries"] == 0


def test_stored_entry_is_not_mutated_by_later_generations(runner):
    run(runner, [1, 2, 3, 4])
    run(runner, [1, 2, 3, 4, 100, 101, 5])
    # Re-running the second prompt must still hit on its own stored entry.
    out = run(runner, [1, 2, 3, 4, 100, 101, 5, 100, 101, 6])
    assert fake_stream_generate.calls[-1] == [6]
    assert out[0].cached_tokens == 9


def test_cache_disabled_passes_prompt_through_untouched(monkeypatch):
    fake_stream_generate.calls = []
    monkeypatch.setattr(mlx_lm, "stream_generate",
                        lambda m, t, p, max_tokens=1, **kw: iter([Resp(1, len(p), "stop")]))
    r = TextRunner(Model(), Tokenizer())
    out = run(r, [1, 2, 3])
    assert r.cache_stats() is None
    assert out[0].cached_tokens == 0 and out[0].prompt_tokens == 3
