import sys

sys.path.insert(0, ".")
from tokenpacker.jitter import predict_itl_jitter
from tokenpacker.steps import compute_steps


def test_jitter_comparison():
    val_chunked = predict_itl_jitter([1024], 512, 64, unchunked=False)
    val_unchunked = predict_itl_jitter([1024], 512, 64, unchunked=True)
    assert val_unchunked > val_chunked


def test_compute_steps_positive():
    assert compute_steps(256, 128, 16) > 0
