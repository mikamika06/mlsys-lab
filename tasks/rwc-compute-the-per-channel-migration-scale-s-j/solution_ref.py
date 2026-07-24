import numpy as np

def compute_migration_scales(W: np.ndarray, X: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute per‑channel migration scales.

    Parameters
    ----------
    W : np.ndarray
        Weight tensor of shape (C_out, *).
    X : np.ndarray
        Activation tensor of shape (N, C_out, *).
    alpha : float
        Hyper‑parameter in [0, 1].

    Returns
    -------
    s : np.ndarray
        One‑dimensional array of length C_out containing the scales.
    """
    out_c = W.shape[0]
    # Max over all elements in each output channel of the weight tensor
    max_W = np.max(np.abs(W.reshape(out_c, -1)), axis=1)
    # Max over batch and spatial dimensions for each activation channel
    max_X = np.max(np.abs(X.reshape(X.shape[0], X.shape[1], -1)), axis=(0, 2))
    return (max_X ** alpha) / (max_W ** (1 - alpha))
