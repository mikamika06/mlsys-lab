import numpy as np

def mask_banned_tokens(logits: np.ndarray, banned_indices: list[int]) -> np.ndarray:
    out = logits.copy()
    if out.ndim == 1:
        out[banned_indices] = -np.inf
    else:
        out[:, banned_indices] = -np.inf
    return out
