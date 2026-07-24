import numpy as np


def branchless_topk_mask(logits: np.ndarray, k: int) -> np.ndarray:
    """Return a copy of `logits` where only the top‑k entries are kept,
    others set to -inf, without any Python branching."""
    raise NotImplementedError("Implement branchless top‑k mask here.")
