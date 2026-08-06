import numpy as np
from lora.forward import lora_forward


def test_lora_scaling_invariant():
    np.random.seed(42)
    x = np.random.randn(4, 32)
    w = np.random.randn(16, 32)
    a = np.random.randn(2, 32)
    b = np.random.randn(16, 2)
    alpha = 8.0
    rank = 2
    out = lora_forward(x, w, a, b, alpha, rank)
    scaling = alpha / float(rank)
    expected = np.matmul(x, w.T) + scaling * np.matmul(np.matmul(x, a.T), b.T)
    assert np.allclose(out, expected, atol=1e-5, rtol=1e-5)
    out_diff_alpha = lora_forward(x, w, a, b, alpha=4.0, rank=rank)
    assert not np.allclose(out, out_diff_alpha, atol=1e-5, rtol=1e-5)
