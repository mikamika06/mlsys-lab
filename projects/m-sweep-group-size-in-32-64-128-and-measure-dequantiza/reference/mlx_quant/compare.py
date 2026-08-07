"""Comparison of 4-bit vs 8-bit model footprint and drift."""

import numpy as np
from mlx_quant.sweep import dequantize_affine, quantize_affine
from mlx_quant.unpack import pack_uint4_pair


def compare_bit_widths(weights: np.ndarray, group_size: int = 64) -> dict:
    """Compare size bytes and dequantization MSE for 4-bit vs 8-bit quantization."""
    w_fp32 = weights.astype(np.float32)
    n_elems = w_fp32.size
    num_groups = (n_elems + group_size - 1) // group_size

    q4, s4, b4 = quantize_affine(w_fp32, group_size=group_size, bits=4)
    deq4 = dequantize_affine(q4, s4, b4, group_size=group_size)
    mse4 = float(np.mean((w_fp32 - deq4) ** 2))
    packed_q4 = pack_uint4_pair(q4)
    bytes4 = packed_q4.nbytes + s4.nbytes + b4.nbytes

    q8, s8, b8 = quantize_affine(w_fp32, group_size=group_size, bits=8)
    deq8 = dequantize_affine(q8, s8, b8, group_size=group_size)
    mse8 = float(np.mean((w_fp32 - deq8) ** 2))
    bytes8 = q8.nbytes + s8.nbytes + b8.nbytes

    fp32_bytes = w_fp32.nbytes

    return {
        "4bit": {
            "bytes": int(bytes4),
            "mse": mse4,
            "compression_ratio": float(fp32_bytes / bytes4),
        },
        "8bit": {
            "bytes": int(bytes8),
            "mse": mse8,
            "compression_ratio": float(fp32_bytes / bytes8),
        },
    }
