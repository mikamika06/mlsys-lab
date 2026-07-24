import numpy as np


def emit_lse(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    row_max = np.max(S, axis=1, keepdims=True)
    return (row_max + np.log(np.sum(np.exp(S - row_max), axis=1, keepdims=True))).ravel()
