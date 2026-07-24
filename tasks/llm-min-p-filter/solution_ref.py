import numpy as np

def minp_filter(probs: np.ndarray, min_p: float) -> np.ndarray:
    """
    Return a boolean mask indicating which tokens have probability at least
    `min_p` times the maximum probability in `probs`.
    """
    max_prob = probs.max()
    threshold = min_p * max_prob
    return probs >= threshold
