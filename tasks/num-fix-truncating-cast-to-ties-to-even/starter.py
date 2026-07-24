import numpy as np


def cast_f32_rne(values: np.ndarray) -> np.ndarray:
    """Convert float64 values to float32 with truncating mantissa bits."""
    x = np.asarray(values, dtype=np.float64)
    out = x.astype(np.float32).view(np.uint32)
    mask = np.uint32(0xFFFFE000)
    out = out & mask
    return out.view(np.float32)
