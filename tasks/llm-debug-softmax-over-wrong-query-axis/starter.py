import numpy as np

def sdpa(query: np.ndarray,
         key: np.ndarray,
         value: np.ndarray,
         scale: float | None = None) -> np.ndarray:
    """
    TODO: This implementation mistakenly applies softmax over the query axis.
    It should be applied over the key dimension (last axis of scores).
    """
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("All inputs must be 3‑D arrays.")
    B, Nq, dk = query.shape
    _, Nk, _ = key.shape
    _, _, dv = value.shape

    if key.shape[2] != dk or value.shape[1] != Nk:
        raise ValueError("Incompatible shapes.")

    if scale is None:
        scale = 1.0 / np.sqrt(dk)

    scores = query @ key.transpose(0, 2, 1) * scale
    # WRONG: softmax over the query dimension (axis=-2)
    exp_scores = np.exp(scores - np.max(scores, axis=-2, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=-2, keepdims=True)

    return probs @ value
