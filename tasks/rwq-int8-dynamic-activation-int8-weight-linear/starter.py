import numpy as np

def int8_linear(X, W):
    """INT8 quantized linear: per-channel symmetric W, per-token symmetric X.

    X: float64 array [B, K]
    W: float64 array [N, K]
    Returns: float64 array [B, N]
    """
    raise NotImplementedError("your code here")
