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
    n_samples, n_channels = X.shape
    scales = []
    for j in range(n_channels):
        max_x = abs(X[0, j])
        for i in range(1, n_samples):
            val = abs(X[i, j])
            if val > max_x:
                max_x = val
        
        max_w = abs(W[0, j])
        for i in range(1, n_samples):
            val = abs(W[i, j])
            if val > max_w:
                max_w = val
        
        scale = (max_x ** alpha) / (max_w ** (1.0 - alpha))
        scales.append(scale)
        
    return np.array(scales, dtype=np.float64)
