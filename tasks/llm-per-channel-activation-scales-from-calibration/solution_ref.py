import numpy as np

def per_channel_scales(X: np.ndarray) -> np.ndarray:
    """
    Compute the per‑channel RMS scale for a calibration tensor X.

    Parameters
    ----------
    X : np.ndarray
        2‑D array of shape (N, C).

    Returns
    -------
    scales : np.ndarray
        1‑D float64 array of length C containing the RMS magnitude of each channel.
    """
    X = np.asarray(X, dtype=np.float64)
    # Root‑mean‑square over the sample dimension
    return np.linalg.norm(X, axis=0) / np.sqrt(X.shape[0])
