"""Blockwise 4-bit quantization and dequantization."""

import numpy as np


def quantize_blockwise(x, block_size, codebook):
    """Quantize array blockwise using max absolute scale per block."""
    raise NotImplementedError


def dequantize_blockwise(q_indices, scales, block_size, codebook):
    """Dequantize 4-bit indices back to float using block scales and codebook."""
    raise NotImplementedError
