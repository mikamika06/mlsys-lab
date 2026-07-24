import numpy as np


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    cutoff = np.searchsorted(cumulative, p, side="left")
    return order[: cutoff + 1]
