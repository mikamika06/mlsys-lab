import numpy as np
from awqsim.scale import compute_awq_scales


def test_exploding_scale_guard():
    np.random.seed(123)
    X = np.random.randn(32, 64)
    X[:, 0] *= 200.0
    W = np.random.randn(64, 64)

    ratio = 4.0
    scales = compute_awq_scales([W], X, alpha=0.5, max_scale_ratio=ratio)

    med = np.median(scales)
    max_allowed = ratio * med
    assert np.max(scales) <= max_allowed + 1e-5
