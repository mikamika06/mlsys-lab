import numpy as np

def compute_activation_scale(X: np.ndarray) -> np.ndarray:
    """
    Compute per‑channel mean absolute activation.

    Parameters
    ----------
    X : np.ndarray
        3‑D array of shape (batch, seq_len, channels).

    Returns
    -------
    np.ndarray
        1‑D float64 array of length `channels` containing the statistic.
    """
    # Use vectorised NumPy; axis=(0,1) collapses batch and sequence dimensions
    return np.mean(np.abs(X), axis=(0, 1)).astype(np.float64)
