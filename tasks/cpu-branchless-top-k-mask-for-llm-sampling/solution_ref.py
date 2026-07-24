import numpy as np


def branchless_topk_mask(logits: np.ndarray, k: int) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.size
    if k <= 0:
        return np.full_like(logits, -np.inf)
    # find threshold through partial sort
    tau = np.partition(logits, -k)[-k]
    # branchless mask: np.where
    mask = np.where(logits >= tau, logits, -np.inf)
    return mask
