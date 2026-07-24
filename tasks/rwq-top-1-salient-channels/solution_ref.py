import numpy as np


def top_salient_channels(X: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """AWQ-style salient-channel selection: the top `frac` fraction of
    channels (columns) by mean absolute calibration activation.

    Returns a 1-D int array of channel indices, length ceil(frac * C)
    (minimum 1), the channels with the largest mean(|X|).
    """
    X = np.asarray(X, dtype=np.float64)
    C = X.shape[1]
    scores = np.mean(np.abs(X), axis=0)
    k = max(1, int(np.ceil(frac * C)))
    order = np.argsort(-scores, kind="stable")
    return order[:k].astype(np.int64)
