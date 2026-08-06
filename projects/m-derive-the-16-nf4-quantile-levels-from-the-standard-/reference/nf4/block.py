import numpy as np
from nf4.quant import get_nf4_quantiles


def quantize_nf4_blockwise(x, block_size=64):
    x_flat = x.flatten().astype(np.float64)
    pad_len = (block_size - (len(x_flat) % block_size)) % block_size
    if pad_len > 0:
        x_padded = np.pad(x_flat, (0, pad_len), mode="constant")
    else:
        x_padded = x_flat

    blocks = x_padded.reshape(-1, block_size)
    absmax = np.max(np.abs(blocks), axis=1, keepdims=True)
    absmax = np.maximum(absmax, 1e-12)

    normalized = blocks / absmax
    quantiles = get_nf4_quantiles()

    indices = np.abs(normalized[:, :, None] - quantiles[None, None, :]).argmin(axis=2)
    dequantized_blocks = quantiles[indices] * absmax

    dequantized_flat = dequantized_blocks.flatten()
    if pad_len > 0:
        dequantized_flat = dequantized_flat[:len(x_flat)]

    return dequantized_flat.reshape(x.shape)
