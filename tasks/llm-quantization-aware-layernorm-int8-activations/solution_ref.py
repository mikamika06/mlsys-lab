import math
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
    B, F = x.shape
    y = np.empty((B, F), dtype=np.float64)

    for b in range(B):
        sum_x = 0.0
        for f in range(F):
            sum_x += float(x[b, f])
        mu = sum_x / F

        sum_sq = 0.0
        for f in range(F):
            diff = float(x[b, f]) - mu
            sum_sq += diff * diff
        var = sum_sq / F

        denom = math.sqrt(var + eps)

        for f in range(F):
            y[b, f] = (float(x[b, f]) - mu) / denom * float(gamma[f]) + float(beta[f])

    return y
