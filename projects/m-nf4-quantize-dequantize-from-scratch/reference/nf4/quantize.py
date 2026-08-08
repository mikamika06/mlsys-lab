import numpy as np


def quantize_blockwise(tensor, codebook, block_size=64):
    N = len(tensor)
    num_blocks = N // block_size
    tensor = tensor.reshape(num_blocks, block_size)

    absmax = np.max(np.abs(tensor), axis=1)
    scale = np.where(absmax == 0, 1.0, absmax)

    normalized = tensor / scale[:, None]

    diffs = np.abs(normalized[..., None] - codebook)
    indices = np.argmin(diffs, axis=-1).astype(np.uint8)
    indices = indices.reshape(-1)

    high = indices[0::2] << 4
    low = indices[1::2]
    quantized = high | low

    return quantized, absmax


def dequantize_blockwise(quantized, absmax, codebook, block_size=64):
    high = (quantized >> 4) & 0x0F
    low = quantized & 0x0F

    indices = np.empty(len(quantized) * 2, dtype=np.uint8)
    indices[0::2] = high
    indices[1::2] = low

    values = codebook[indices]
    values = values.reshape(-1, block_size)
    values = values * absmax[:, None]

    return values.reshape(-1)
