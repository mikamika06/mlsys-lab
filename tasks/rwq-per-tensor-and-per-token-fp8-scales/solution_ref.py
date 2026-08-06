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
    max_w = -float('inf')
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            val = abs(W[i, j])
            if val > max_w:
                max_w = val
    tensor_scale = max_w / 448.0

    if X.ndim == 2:
        X_2d = X
    else:
        X_2d = X.reshape(-1, X.shape[-1])

    token_scales_list = []
    for i in range(X_2d.shape[0]):
        max_x = -float('inf')
        for j in range(X_2d.shape[1]):
            val = abs(X_2d[i, j])
            if val > max_x:
                max_x = val
        token_scales_list.append(max_x / 448.0)

    token_scales = np.array(token_scales_list, dtype=X.dtype)
    return tensor_scale, token_scales
