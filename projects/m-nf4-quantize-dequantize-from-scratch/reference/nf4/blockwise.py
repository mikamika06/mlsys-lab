import numpy as np


def quantize_blockwise(x: np.ndarray, codebook: np.ndarray, block_size: int = 64):
    """Quantize array into 4-bit indices (0-15) and per-block absmax scale factors."""
    orig_shape = x.shape
    flat = x.astype(np.float64).flatten()
    n = len(flat)

    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        flat_padded = np.pad(flat, (0, pad_len), mode="constant", constant_values=0.0)
    else:
        flat_padded = flat

    blocks = flat_padded.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales[scales == 0.0] = 1.0

    norm_blocks = blocks / scales
    diffs = np.abs(norm_blocks[:, :, None] - codebook[None, None, :])
    indices = np.argmin(diffs, axis=2).astype(np.uint8)

    return indices, scales.squeeze(axis=1), orig_shape


def dequantize_blockwise(indices: np.ndarray, scales: np.ndarray, codebook: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Dequantize 4-bit indices and scale factors back to float representation."""
    block_size = indices.shape[1]
    dequant = codebook[indices] * scales[:, None]
    flat = dequant.flatten()
    total_elements = int(np.prod(original_shape))
    return flat[:total_elements].reshape(original_shape)
