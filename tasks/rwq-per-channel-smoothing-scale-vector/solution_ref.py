import numpy as np

def per_channel_scale(X: np.ndarray, W: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute per‑channel smoothing scale vector.

    Parameters
    ----------
    X : np.ndarray
        Input activations of shape (n_samples, n_channels).
    W : np.ndarray
        Reference weights of the same shape as `X`.
    alpha : float
        Trade‑off parameter in [0, 1].

    Returns
    -------
    s : np.ndarray
        Scale vector of shape (n_channels,) with dtype float64.
    """
    max_X = np.max(np.abs(X), axis=0)
    max_W = np.max(np.abs(W), axis=0)
    return (max_X ** alpha) / (max_W ** (1 - alpha))
