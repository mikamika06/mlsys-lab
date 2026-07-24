import numpy as np

def logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Compute the log‑sum‑exp of `x` along `axis` with numerical stability."""
    x = np.asarray(x, dtype=np.float64)
    if axis is None:
        max_val = np.max(x)
        return np.log(np.sum(np.exp(x - max_val))) + max_val
    else:
        max_val = np.max(x, axis=axis, keepdims=True)
        sum_exp = np.sum(np.exp(x - max_val), axis=axis)
        return np.squeeze(np.log(sum_exp) + np.squeeze(max_val))
