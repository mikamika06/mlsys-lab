import numpy as np


def fold_scales(X: np.ndarray, W: np.ndarray, scales: np.ndarray):
    """Folds per-channel scales s into X and W such that (X / s) @ (s * W) == X @ W."""
    X_scaled = X / scales
    W_scaled = W * scales[:, None]
    return X_scaled, W_scaled


def quantize_per_tensor(W: np.ndarray, n_bits: int = 4):
    """Uniform symmetric per-tensor min-max quantization/dequantization."""
    qmin = -(2 ** (n_bits - 1))
    qmax = 2 ** (n_bits - 1) - 1
    max_val = np.max(np.abs(W))
    if max_val == 0:
        return W.copy()
    scale = max_val / qmax
    W_q = np.clip(np.round(W / scale), qmin, qmax)
    return W_q * scale
