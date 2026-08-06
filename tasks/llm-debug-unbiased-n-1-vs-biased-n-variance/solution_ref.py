import numpy as np

def layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply Layer Normalization to a 2‑D array using unbiased variance.

    Parameters
    ----------
    x : np.ndarray of shape (n, d)
        Input activations.
    eps : float, optional
        Small constant added to the denominator for numerical stability.

    Returns
    -------
    y : np.ndarray of shape (n, d)
        Normalised activations.  The output has dtype float64.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True, ddof=1)
    return (x - mean) / np.sqrt(var + eps)
