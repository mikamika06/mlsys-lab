import numpy as np

def minp_filter(probs: np.ndarray, min_p: float) -> np.ndarray:
    """
    Return a boolean mask indicating which tokens have probability at least
    `min_p` times the maximum probability in `probs`.
    """
    max_prob = probs.flat[0]
    for x in probs.flat:
        if x > max_prob:
            max_prob = x
    threshold = min_p * max_prob

    out = np.empty(probs.shape, dtype=bool)
    for i in range(out.size):
        out.flat[i] = probs.flat[i] >= threshold
    return out
