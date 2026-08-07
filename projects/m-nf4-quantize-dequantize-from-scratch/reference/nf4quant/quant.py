import numpy as np
from nf4quant.codebook import get_nf4_codebook


def quantize_dequantize_nf4(weights: np.ndarray, block_size: int = 64) -> np.ndarray:
    """Quantize and dequantize weights using blockwise NF4."""
    cb = get_nf4_codebook()
    flat = weights.flatten()
    n = len(flat)
    pad = (block_size - (n % block_size)) % block_size
    if pad > 0:
        flat = np.pad(flat, (0, pad), mode='constant')
    blocks = flat.reshape(-1, block_size)
    out_blocks = np.empty_like(blocks)
    for i in range(blocks.shape[0]):
        b = blocks[i]
        mx = np.max(np.abs(b))
        if mx == 0:
            out_blocks[i] = np.zeros_like(b)
            continue
        norm = b / mx
        idx = np.argmin(np.abs(norm[:, None] - cb[None, :]), axis=1)
        out_blocks[i] = cb[idx] * mx
    res = out_blocks.flatten()
    if pad > 0:
        res = res[:n]
    return res.reshape(weights.shape)
