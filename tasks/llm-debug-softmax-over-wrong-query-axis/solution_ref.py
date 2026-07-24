import numpy as np

def sdpa(query: np.ndarray,
         key: np.ndarray,
         value: np.ndarray,
         scale: float | None = None) -> np.ndarray:
    """
    Scaled dot‑product attention with correct softmax axis.

    Parameters
    ----------
    query : np.ndarray
        Shape (B, N_q, d_k)
    key : np.ndarray
        Shape (B, N_k, d_k)
    value : np.ndarray
        Shape (B, N_k, d_v)
    scale : float | None, optional
        Scaling factor. If None, defaults to 1/sqrt(d_k).

    Returns
    -------
    np.ndarray
        Attention output of shape (B, N_q, d_v).
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
    # softmax over the key dimension (last axis)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return probs @ value
