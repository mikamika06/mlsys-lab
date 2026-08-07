import numpy as np
import pytest
from mxquant.quant import quantize_per_group, quantize_linear


def test_group_size_divisibility_raises():
    w = np.random.randn(16, 32)
    with pytest.raises(ValueError):
        quantize_per_group(w, group_size=10, bits=4)


def test_quantize_linear_compression_ratio():
    w = np.random.randn(64, 128)
    _, _, _, ratio = quantize_linear(w, group_size=64, bits=4)
    assert ratio > 1.0


def test_scale_bias_shapes():
    w = np.random.randn(32, 64)
    _, s, b = quantize_per_group(w, group_size=32, bits=4)
    assert s.shape == (32, 2, 1)
    assert b.shape == (32, 2, 1)
