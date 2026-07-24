import numpy as np

def top_k_filter(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a boolean mask of the indices belonging to the top‑k logits and
    the filtered logits where all other entries are set to -inf.
    """
    sorted_idx = np.argsort(-logits)
    mask = np.zeros_like(logits, dtype=bool)
    mask[sorted_idx[:k]] = True

    filtered = logits.copy()
    filtered[~mask] = -np.inf
    return mask, filtered
