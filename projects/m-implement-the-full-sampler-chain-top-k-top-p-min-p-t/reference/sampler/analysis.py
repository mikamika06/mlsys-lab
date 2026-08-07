import numpy as np


def compare_min_p_top_p(logits: np.ndarray, min_p: float = 0.05, top_p: float = 0.9) -> dict[str, list[int]]:
    """Derives surviving token indices under min_p vs top_p truncation."""
    shift = logits - np.max(logits)
    probs = np.exp(shift)
    probs /= np.sum(probs)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cum_probs = np.cumsum(sorted_probs)
    top_p_cutoff = np.searchsorted(cum_probs, top_p)
    top_p_cutoff = min(top_p_cutoff, len(sorted_indices) - 1)
    top_p_survivors = sorted(sorted_indices[:top_p_cutoff + 1].tolist())

    max_p = np.max(probs)
    min_p_threshold = min_p * max_p
    min_p_survivors = sorted(np.where(probs >= min_p_threshold)[0].tolist())

    return {
        "top_p_survivors": top_p_survivors,
        "min_p_survivors": min_p_survivors
    }
