"""Regression tests for quantization accuracy and unpack logic."""

import numpy as np
from mlx_quant.sweep import dequantize_affine, quantize_affine
from mlx_quant.unpack import pack_uint4_pair, unpack_and_dequantize_4bit


def test_dequantization_accuracy():
    """Verify dequantization matches scale and bias reconstruction."""
    rng = np.random.RandomState(42)
    weights = rng.randn(128, 64).astype(np.float32)
    group_size = 32

    qw, scales, biases = quantize_affine(weights, group_size=group_size, bits=4)
    deq = dequantize_affine(qw, scales, biases, group_size=group_size)

    max_err = float(np.max(np.abs(weights - deq)))
    assert max_err < 0.5, f"Dequantization error too large: {max_err}"


def test_packed_unpack_alignment():
    """Verify unpacked 4-bit match original unpacked values."""
    rng = np.random.RandomState(42)
    weights = rng.randn(64, 64).astype(np.float32)
    group_size = 32

    qw, scales, biases = quantize_affine(weights, group_size=group_size, bits=4)
    packed = pack_uint4_pair(qw)
    unpacked_deq = unpack_and_dequantize_4bit(
        packed, scales, biases, group_size, original_shape=weights.shape
    )
    direct_deq = dequantize_affine(qw, scales, biases, group_size)

    np.testing.assert_allclose(
        unpacked_deq, direct_deq, rtol=1e-5, atol=1e-5, err_msg="Unpacked dequantization mismatch"
    )
