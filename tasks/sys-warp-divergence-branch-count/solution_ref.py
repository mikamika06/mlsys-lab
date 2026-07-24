import numpy as np

def warp_divergence_branch_count(preds: np.ndarray, warp_size: int = 32) -> np.ndarray:
    preds = np.asarray(preds)
    if preds.ndim != 1:
        raise ValueError("preds must be a one‑dimensional array")
    n = len(preds)
    if n % warp_size != 0:
        raise ValueError(f"Length {n} is not a multiple of warp_size {warp_size}")
    reshaped = preds.reshape(-1, warp_size)
    out = np.empty(reshaped.shape[0], dtype=int)
    for i, block in enumerate(reshaped):
        out[i] = len(np.unique(block))
    return out
