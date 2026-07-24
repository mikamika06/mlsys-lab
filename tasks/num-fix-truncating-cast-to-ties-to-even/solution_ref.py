import numpy as np


def cast_f32_rne(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).astype(np.float32)
