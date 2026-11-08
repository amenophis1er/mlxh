"""TextRunner prefix cache: reuse, exact-hit step-back, bounds, and stats."""
import copy

import pytest

import mlx_lm
from mlx_lm.models import cache as cache_mod

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
    """Hashable stand-in; the trie keys entries by model."""

    def __init__(self, trimmable=True):
        self.trimmable = trimmable


class Tokenizer:
    def decode(self, ids):
        return "".join(chr(97 + i % 26) for i in ids)


class Resp:
    def __init__(self, token, prompt_tokens, finish_reason=None):
        self.token = token
        self.text = chr(97 + token % 26)
        self.prompt_tokens = prompt_tokens
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


def test_non_trimmable_cache_only_reuses_exact_stored_prefixes(monkeypatch):
    fake_stream_generate.calls = []
    monkeypatch.setattr(mlx_lm, "stream_generate", fake_stream_generate)
    monkeypatch.setattr(cache_mod, "make_prompt_cache",
                        lambda model: [FakeLinear(), FakeKV()])
    r = TextRunner(Model(trimmable=False), Tokenizer())
    r.enable_cache(max_bytes=10**9)
    run(r, [1, 2, 3, 4])
    # Divergent suffix: cannot trim, so the whole prompt is reprocessed.
    run(r, [1, 2, 3, 9])
    assert fake_stream_generate.calls[-1] == [1, 2, 3, 9]
    # Exact hit on a non-trimmable cache cannot step back: full reprocess.
    run(r, [1, 2, 3, 4, 100, 101])
    assert fake_stream_generate.calls[-1] == [1, 2, 3, 4, 100, 101]
    # A stored entry that is a strict prefix is still reused.
    run(r, [1, 2, 3, 4, 100, 101, 7])
    assert fake_stream_generate.calls[-1] == [7]


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
