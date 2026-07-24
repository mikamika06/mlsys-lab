import numpy as np


def fused_affine(X: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # TODO: fused kernel with an incorrect broadcast dimension.
    # It applies scale and bias using the row index instead of the column index.
    X = np.asarray(X, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    n, d = X.shape
    out = np.empty_like(X)
    for i in range(n):
        out[i] = X[i] * scale[i % len(scale)] + bias[i % len(bias)]
    return out
