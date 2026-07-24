import numpy as np
from scipy.stats import norm


def nf4_levels() -> np.ndarray:
    """The 16 NF4 codebook levels: equal-probability-mass quantiles of N(0,1),
    asymmetric (8 non-negative incl. exact 0, 7 negative), normalized to [-1, 1]."""
    offset = 0.9677083
    v_pos = norm.ppf(np.linspace(offset, 0.5, 9)[:-1])       # 8 non-negative
    v_neg = -norm.ppf(np.linspace(offset, 0.5, 8)[:-1])       # 7 negative
    v = np.concatenate([v_pos, np.array([0.0]), v_neg])
    v = np.sort(v)
    v = v / np.max(np.abs(v))
    return v.astype(np.float64)


def quantize_4bit(x: np.ndarray, block_size: int = 64) -> tuple:
    """Blockwise absmax NF4 quantization: nearest-codebook-level 4-bit codes,
    packed two-per-byte, plus one float32 scale per block."""
    levels = nf4_levels()
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    n_blocks = int(np.ceil(n / block_size))

    codes = np.zeros(n, dtype=np.uint8)
    absmax = np.zeros(n_blocks, dtype=np.float32)
    for b in range(n_blocks):
        s, e = b * block_size, min((b + 1) * block_size, n)
        block = x[s:e]
        am = float(np.max(np.abs(block)))
        if am == 0.0:
            am = 1.0
        absmax[b] = am
        norm_block = block / am
        d = np.abs(norm_block[:, None] - levels[None, :])
        codes[s:e] = np.argmin(d, axis=1).astype(np.uint8)

    n_packed = (n + 1) // 2
    packed = np.zeros(n_packed, dtype=np.uint8)
    even = codes[0::2]
    odd = codes[1::2]
    packed[:even.shape[0]] |= even
    packed[:odd.shape[0]] |= (odd << 4)
    return packed, absmax


def dequantize_4bit(packed: np.ndarray, absmax: np.ndarray, n: int, block_size: int = 64) -> np.ndarray:
    """Inverse of quantize_4bit: unpack nibbles, look up codebook level, scale by block absmax."""
    levels = nf4_levels()
    packed = np.asarray(packed, dtype=np.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    codes = np.zeros(n, dtype=np.uint8)
    idx_even = np.arange(0, n, 2)
    idx_odd = np.arange(1, n, 2)
    codes[idx_even] = low[:idx_even.shape[0]]
    codes[idx_odd] = high[:idx_odd.shape[0]]

    out = np.zeros(n, dtype=np.float64)
    n_blocks = int(np.ceil(n / block_size))
    for b in range(n_blocks):
        s, e = b * block_size, min((b + 1) * block_size, n)
        out[s:e] = levels[codes[s:e]] * absmax[b]
    return out
