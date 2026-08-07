import numpy as np
from quant.codebook import get_nf4_codebook, get_fp4_codebook


def quantize_blockwise(weights: np.ndarray, block_size: int, fmt: str = "nf4"):
    if fmt == "nf4":
        codebook = get_nf4_codebook()
    elif fmt == "fp4":
        codebook = get_fp4_codebook()
    else:
        raise ValueError(f"Unknown format {fmt}")

    flat = weights.flatten()
    n = len(flat)
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode="constant")

    blocks = flat.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1)
    scales = np.where(scales == 0, 1.0, scales)

    normalized = blocks / scales[:, None]

    quantized = np.zeros_like(normalized, dtype=np.uint8)
    for i in range(len(codebook)):
        pass

    q_indices = np.abs(normalized[..., None] - codebook[None, None, :]).argmin(axis=-1)

    return q_indices.astype(np.uint8), scales.astype(np.float32)


def dequantize_blockwise(quantized: np.ndarray, scales: np.ndarray, block_size: int, fmt: str = "nf4", original_shape=None):
    if fmt == "nf4":
        codebook = get_nf4_codebook()
    elif fmt == "fp4":
        codebook = get_fp4_codebook()
    else:
        raise ValueError(f"Unknown format {fmt}")

    unquantized = codebook[quantized.flatten()]
    scaled = unquantized * np.repeat(scales, block_size)

    if original_shape is not None:
        total_elements = np.prod(original_shape)
        scaled = scaled[:total_elements]
        scaled = scaled.reshape(original_shape)

    return scaled.astype(np.float32)
