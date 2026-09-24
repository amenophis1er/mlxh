"""Opt-in release check: MLXH_TEST_IMAGE_MODEL=/path/to/pack pytest ..."""

import os

import pytest


@pytest.mark.skipif(not os.environ.get("MLXH_TEST_IMAGE_MODEL"), reason="requires downloaded image weights")
def test_real_image_engine():
    from mlxh.image_engine import ImageEngine
    from mlxh.images import ImageError, ImageGenerationResult, parse_request
    from mlxh.serve_app import SETTINGS

    settings = {**SETTINGS, "cache_limit_gb": 1, "memory_limit_gb": 24}
    engine = ImageEngine(os.environ["MLXH_TEST_IMAGE_MODEL"], "schnell", settings)
    engine.start()
    engine.wait_ready()

    def generate(prompt="a red canoe", **overrides):
        req = parse_request({"model": "schnell", "prompt": prompt,
                             "seed": 42, "size": "256x256", **overrides}, "schnell", settings)
        return engine.submit(req).out.get(timeout=180)

    try:
        first = generate()
        assert isinstance(first, ImageGenerationResult)
        assert first.image == generate().image
        assert generate("blue " * 400).status_code == 400
        # Beyond CLIP's 77-token window but within T5's 256-token budget.
        assert isinstance(generate("blue " * 100), ImageGenerationResult)
        settings["gen_timeout_s"] = .001
        result = generate("a yellow canoe")
        assert isinstance(result, ImageError) and result.status_code == 504
        assert engine.snapshot()["runtime"]["last_request"]["outcome"] == "timed_out"
        settings["gen_timeout_s"] = 600
        assert isinstance(generate(), ImageGenerationResult)
        assert not engine.snapshot()["runtime"]["busy"]
    finally:
        engine.stop()
    assert not engine._thread.is_alive()
