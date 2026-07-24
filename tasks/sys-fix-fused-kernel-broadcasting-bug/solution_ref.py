import numpy as np


def fused_affine(X: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    return X * scale + bias
