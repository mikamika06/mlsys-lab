"""Quantization error evaluation across distributions."""

import numpy as np

from .codebook import create_fp4_codebook, create_int4_codebook, create_nf4_codebook
from .quant import dequantize_blockwise, quantize_blockwise


def quantization_error(x, block_size, codebook):
    """Compute MSE reconstruction error for blockwise quantization."""
    q_idx, scales = quantize_blockwise(x, block_size, codebook)
    rec = dequantize_blockwise(q_idx, scales, block_size, codebook)
    return float(np.mean((x - rec) ** 2))


def compare_codebooks_on_distributions(distributions, block_size):
    """Compare NF4, FP4, and INT4 quantization error on multiple distributions."""
    cb_nf4 = create_nf4_codebook()
    cb_fp4 = create_fp4_codebook()
    cb_int4 = create_int4_codebook()

    res = {}
    for name, data in distributions.items():
        res[name] = {
            "nf4": quantization_error(data, block_size, cb_nf4),
            "fp4": quantization_error(data, block_size, cb_fp4),
            "int4": quantization_error(data, block_size, cb_int4),
        }
    return res
