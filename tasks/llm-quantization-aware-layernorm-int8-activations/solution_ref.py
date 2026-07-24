import numpy as np

def layernorm_int8(x: np.ndarray,
                   gamma: np.ndarray,
                   beta: np.ndarray,
                   eps: float = 1e-5) -> np.ndarray:
    """
    Correct implementation of LayerNorm for int8 activations.
    Dequantises the input, normalises across features, and returns
    a float64 array.
    """
    xf = x.astype(np.float64)
    mu = np.mean(xf, axis=-1, keepdims=True)
    var = np.var(xf, axis=-1, keepdims=True)
    denom = np.sqrt(var + eps)
    y = (xf - mu) / denom * gamma + beta
    return y
