import numpy as np

def fp8_scales(W: np.ndarray, X: np.ndarray):
    """
    Compute per‑tensor and per‑token FP8 scales.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (out_dim, in_dim).
    X : np.ndarray
        Activation tensor. Tokens are rows along all axes except the last one.

    Returns
    -------
    tuple[float, np.ndarray]
        Per‑tensor scale and per‑token scales.
    """
    # per‑tensor scale
    tensor_scale = np.max(np.abs(W)) / 448.0

    # compute token max over feature dimension
    if X.ndim == 2:
        token_max = np.max(np.abs(X), axis=1)
    else:
        # collapse all but last axis into one dimension of tokens
        token_max = np.max(np.abs(X.reshape(-1, X.shape[-1])), axis=1)

    token_scales = token_max / 448.0
    return tensor_scale, token_scales
