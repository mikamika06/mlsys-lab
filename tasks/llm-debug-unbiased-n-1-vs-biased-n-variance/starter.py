import numpy as np

def layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    TODO: This implementation uses the biased variance (ddof=0),
    which is incorrect for this task.  Replace it with an unbiased
    variance calculation.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True, ddof=0)  # WRONG: biased variance
    return (x - mean) / np.sqrt(var + eps)
