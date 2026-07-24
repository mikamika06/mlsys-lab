import numpy as np


def affine_quant_dequant(x: np.ndarray, bits: int) -> np.ndarray:
    """Correct affine quantize-dequantize using (2^bits - 1) denominator."""
    x = np.asarray(x, dtype=np.float64)
    n_levels = (1 << bits) - 1
    x_min = x.min()
    x_max = x.max()
    scale = (x_max - x_min) / n_levels if x_max != x_min else 1.0
    q = np.clip(np.round((x - x_min) / scale), 0, n_levels)
    return q * scale + x_min
