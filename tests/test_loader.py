import json

from mlxh.loader import (
    TextRunner, VLMRunner, context_window_of, detect_context_window,
    resolve_context_window,
)


def test_detects_flat_max_position_embeddings():
    cfg = {"model_type": "qwen2", "max_position_embeddings": 32768}
    assert detect_context_window(cfg) == 32768


def test_detects_nested_text_config():
    cfg = {"model_type": "gemma4", "text_config": {"max_position_embeddings": 131072}}
    assert detect_context_window(cfg) == 131072


def test_detects_nested_language_config():
    cfg = {"model_type": "somevlm", "language_config": {"max_position_embeddings": 8192}}
    assert detect_context_window(cfg) == 8192


def test_prefers_top_level_over_nested():
    cfg = {"max_position_embeddings": 4096,
           "text_config": {"max_position_embeddings": 32768}}
    assert detect_context_window(cfg) == 4096


def test_detects_alternate_position_keys():
    assert detect_context_window({"n_positions": 2048}) == 2048
    assert detect_context_window({"context_length": 65536}) == 65536


def test_detects_prism_pack_shape():
    cfg = {"model_type": "prism_x", "text_config": {"max_position_embeddings": 262144}}
    assert detect_context_window(cfg) == 262144


def test_ignores_unknown_or_invalid_values():
    assert detect_context_window({}) is None
    assert detect_context_window({"text_config": {}}) is None
    assert detect_context_window({"max_position_embeddings": "32768"}) is None
    assert detect_context_window({"max_position_embeddings": True}) is None
    assert detect_context_window({"max_position_embeddings": 0}) is None
    assert detect_context_window({"max_position_embeddings": -1}) is None


def test_context_window_of_reads_model_directory(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
    assert context_window_of(model) == 32768


def test_context_window_of_survives_missing_or_broken_configs(tmp_path):
    missing = tmp_path / "missing"
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("not json")
    is_dir = tmp_path / "adir"
    (is_dir / "config.json").mkdir(parents=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text(json.dumps({}))
    assert context_window_of(missing) is None
    assert context_window_of(broken) is None
    assert context_window_of(is_dir) is None
    assert context_window_of(empty) is None


def test_runners_hold_context_window():
    assert TextRunner(None, None, context_window=4096).context_window == 4096
    assert VLMRunner(None, None, {}, context_window=8192).context_window == 8192
    assert TextRunner(None, None).context_window is None
    assert VLMRunner(None, None, {}).context_window is None

def test_resolve_context_window_setting_semantics():
    assert resolve_context_window(0, 32768) == 32768      # 0 = detect
    assert resolve_context_window(None, 32768) == 32768   # missing key = detect
    assert resolve_context_window(-1, 32768) is None      # -1 = window off
    assert resolve_context_window(8192, 32768) == 8192    # explicit override wins
    assert resolve_context_window(8192, None) == 8192     # override despite no detection
    assert resolve_context_window(0, None) is None
    assert resolve_context_window(0, 0) is None
    assert resolve_context_window(0, True) is None
