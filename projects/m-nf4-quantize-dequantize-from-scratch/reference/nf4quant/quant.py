"""Blockwise 4-bit quantization and dequantization."""

import numpy as np


def quantize_blockwise(x, block_size, codebook):
    """Quantize array blockwise using max absolute scale per block."""
    orig_shape = x.shape
    x_flat = np.asarray(x, dtype=np.float64).ravel()
    n = len(x_flat)
    num_blocks = (n + block_size - 1) // block_size
    pad_len = num_blocks * block_size
    if n < pad_len:
        x_pad = np.pad(x_flat, (0, pad_len - n))
    else:
        x_pad = x_flat

    blocks = x_pad.reshape(num_blocks, block_size)
    scales = np.max(np.abs(blocks), axis=1)
    scales_safe = np.where(scales == 0, 1.0, scales)

    norm_blocks = blocks / scales_safe[:, None]
    diffs = np.abs(norm_blocks[:, :, None] - codebook[None, None, :])
    q_indices = np.argmin(diffs, axis=2).astype(np.uint8)

    q_flat = q_indices.ravel()[:n]
    return q_flat.reshape(orig_shape), scales


def dequantize_blockwise(q_indices, scales, block_size, codebook):
    """Dequantize 4-bit indices back to float using block scales and codebook."""
    orig_shape = q_indices.shape
    q_flat = np.asarray(q_indices, dtype=np.uint8).ravel()
    n = len(q_flat)
    num_blocks = len(scales)
    pad_len = num_blocks * block_size
    if n < pad_len:
        q_pad = np.pad(q_flat, (0, pad_len - n), mode="constant", constant_values=0)
    else:
        q_pad = q_flat

    blocks = q_pad.reshape(num_blocks, block_size)
    dequant_blocks = codebook[blocks] * scales[:, None]
    dequant_flat = dequant_blocks.ravel()[:n]
    return dequant_flat.reshape(orig_shape)
