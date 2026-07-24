import numpy as np

def rms_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Correct RMSNorm implementation.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (B, D).
    eps : float, optional
        Small constant added to the denominator for numerical stability.
        Default is 1e-5.

    Returns
    -------
    np.ndarray
        Normalized array with the same shape and dtype float64.
    """
    # Ensure input is float64
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms
