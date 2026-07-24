import numpy as np


def log_condition_number(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.abs(1.0 / np.log(x))
