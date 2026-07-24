import numpy as np

def salient_channels(X: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """
    Return indices of the top fraction of channels by mean absolute activation.
    """
    if not (0 <= fraction <= 1):
        raise ValueError("fraction must be in [0, 1]")
    n_channels = X.shape[1]
    k = int(np.ceil(fraction * n_channels))
    if k == 0:
        return np.array([], dtype=np.int64)
    mean_abs = np.mean(np.abs(X), axis=0)
    idx_desc = np.argsort(-mean_abs)
    topk = idx_desc[:k]
    return np.sort(topk).astype(np.int64)
