import numpy as np


def affine_quant_dequant(x: np.ndarray, bits: int) -> np.ndarray:
    """Affine quantize-dequantize — fix the scale denominator bug."""
    x = np.asarray(x, dtype=np.float64)
    # BUG: uses 2^bits instead of 2^bits - 1
    n_levels = (1 << bits)  # <-- off by one: should be (1 << bits) - 1
    x_min = x.min()
    x_max = x.max()
    scale = (x_max - x_min) / n_levels if x_max != x_min else 1.0
    q = np.clip(np.round((x - x_min) / scale), 0, n_levels - 1)
    return q * scale + x_min
