import numpy as np

def unpack_gqa(kv_packed: np.ndarray, n_rep: int) -> np.ndarray:
    """
    Expand a packed KV tensor to the per‑query‑head layout.

    Parameters
    ----------
    kv_packed : np.ndarray
        Packed key/value tensor of shape (H, L, D).
    n_rep : int
        Number of query heads that share each packed head.

    Returns
    -------
    np.ndarray
        Expanded tensor of shape (H * n_rep, L, D) where each packed head is
        repeated exactly `n_rep` times along the first axis.
    """
    if not isinstance(kv_packed, np.ndarray):
        raise TypeError("kv_packed must be a NumPy array")
    if kv_packed.ndim != 3:
        raise ValueError("kv_packed must have shape (H, L, D)")
    if n_rep <= 0 or not isinstance(n_rep, int):
        raise ValueError("n_rep must be a positive integer")

    return np.repeat(kv_packed, n_rep, axis=0)
