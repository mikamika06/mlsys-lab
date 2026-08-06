import numpy as np


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Filter top probabilities up to cumulative sum p."""
    n = len(probs)
    order = sorted(range(n), key=lambda i: probs[i], reverse=True)
    res = []
    cum = 0.0
    for idx in order:
        res.append(idx)
        cum += float(probs[idx])
        if cum >= p:
            break
    return np.array(res, dtype=np.int64)
