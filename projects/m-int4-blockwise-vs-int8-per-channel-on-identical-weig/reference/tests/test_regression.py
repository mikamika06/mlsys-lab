import numpy as np
from quant.core import (
    dequantize_int4_blockwise,
    dequantize_int8_per_channel,
    quantize_int4_blockwise,
    quantize_int8_per_channel,
)
from quant.metrics import compute_bit_size, compute_mse


def test_int8_per_channel_bounds():
    np.random.seed(42)
    w = np.random.randn(16, 64).astype(np.float32)
    q, scales = quantize_int8_per_channel(w)
    assert np.all(q >= -128) and np.all(q <= 127)
    dequant = dequantize_int8_per_channel(q, scales)
    mse = compute_mse(w, dequant)
    assert mse < 0.05


def test_int4_blockwise_reconstruction():
    np.random.seed(42)
    w = np.random.randn(16, 64).astype(np.float32)
    q, scales = quantize_int4_blockwise(w, block_size=16)
    assert np.all(q >= -7) and np.all(q <= 7)
    dequant = dequantize_int4_blockwise(q, scales, block_size=16)
    mse = compute_mse(w, dequant)
    assert mse < 0.1

    bits_int8 = compute_bit_size((16, 64), "int8_per_channel")
    bits_int4 = compute_bit_size((16, 64), "int4_blockwise", block_size=16)
    assert bits_int4 < bits_int8
