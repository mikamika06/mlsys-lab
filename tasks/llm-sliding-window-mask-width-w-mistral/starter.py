import numpy as np


def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, w: int) -> np.ndarray:
    """Single-head scaled dot-product attention with a Mistral sliding-window mask.

    Query i attends only to keys j with i - w < j <= i (the w most recent keys,
    including itself). Masked positions are set to -inf before the softmax.
    Return the (n, d) attention output in float64.
    """
    raise NotImplementedError("your code here")
