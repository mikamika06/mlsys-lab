import numpy as np


def apply_repetition_penalty(logits: np.ndarray, penalty: float) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    out = x.copy()
    out[x > 0] = x[x > 0] / penalty
    out[x < 0] = x[x < 0] * penalty
    return out
