import numpy as np


def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, w: int) -> np.ndarray:
    """Single-head scaled dot-product attention with a Mistral sliding-window mask.

    Query i attends only to keys j with i - w < j <= i (the w most recent keys,
    including itself). Masked positions are set to -inf before the softmax.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, d = Q.shape

    scores = Q @ K.T / np.sqrt(d)

    rows = np.arange(n).reshape(-1, 1)
    cols = np.arange(n).reshape(1, -1)
    allowed = (cols <= rows) & (rows - cols < w)  # i - w < j <= i

    scores = np.where(allowed, scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return p @ V
