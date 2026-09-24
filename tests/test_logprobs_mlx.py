import pytest

from mlxh import serve_app
from mlxh.engine import LogprobsOptions


mx = pytest.importorskip("mlx.core")


def test_capture_normalizes_bfloat16_logits_in_float32():
    captured, processor = serve_app._capture_float32_logprobs(mx)
    logits = mx.array([[1.0, 2.0, 3.0]], dtype=mx.bfloat16)

    returned = processor(mx.array([1]), logits)
    expected_logits = logits.astype(mx.float32)
    expected = expected_logits - mx.logsumexp(
        expected_logits, axis=-1, keepdims=True
    )

    assert returned is logits
    assert captured[0].dtype == mx.float32
    assert mx.allclose(captured[0], expected.squeeze(0)).item()


def test_compute_logprobs_clamps_top_k_and_sorts_targeted_union():
    response = type("Response", (), {
        "token": 1,
    })()
    logprobs = mx.array([-3.0, -0.1, -2.0, -1.0])

    serve_app._compute_logprobs(
        response, logprobs,
        LogprobsOptions(top=128, ids=[2], allowed=[], as_ids=False), mx,
    )

    ids, values, token_lp = response.logprobs_out
    assert ids == [1, 3, 2, 0]
    assert values == sorted(values, reverse=True)
    assert token_lp == pytest.approx(-0.1)
