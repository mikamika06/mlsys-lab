import numpy as np

def quantize_fixed_point(arr: np.ndarray, frac_bits: int) -> np.ndarray:
    """Quantizes a float array to fixed‑point with round‑half‑to‑even."""
    scaled = arr.astype(np.float64) * (1 << frac_bits)
    return np.round(scaled, decimals=0).astype(np.int64)
