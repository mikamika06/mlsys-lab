import numpy as np

def stable_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable log‑softmax.

    Parameters
    ----------
    logits : np.ndarray
        Input array of arbitrary shape.
    axis : int, default -1
        Axis along which to apply the softmax.

    Returns
    -------
    np.ndarray
        Array of same shape and dtype float64 containing the log‑softmax values.
    """
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits, axis=axis, keepdims=True)
    exp_shifted = np.exp(logits - m)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    log_sum_exp = np.log(sum_exp)
    return logits - m - log_sum_exp
