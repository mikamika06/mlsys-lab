import numpy as np


def derive_e4m3_scale(tensor: np.ndarray) -> float:
    """Computes the maximum scale factor to fit tensor into FP8 E4M3 range."""
    tensor = np.asarray(tensor, dtype=np.float32)
    max_abs = float(np.max(np.abs(tensor)))
    if max_abs == 0.0:
        return 1.0
    max_e4m3 = 448.0
    return max_e4m3 / max_abs
