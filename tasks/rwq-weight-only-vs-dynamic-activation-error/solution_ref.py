import numpy as np


def _sym_quant_per_row(A: np.ndarray) -> np.ndarray:
    absmax = np.max(np.abs(A), axis=1, keepdims=True)
    absmax = np.where(absmax == 0.0, 1.0, absmax)
    scale = absmax / 127.0
    codes = np.clip(np.round(A / scale), -127, 127)
    return codes * scale


def weight_only_vs_dynamic_mse(x: np.ndarray, W: np.ndarray):
    """
    Compute a linear layer's output MSE two ways vs the full-precision
    reference y_fp = x @ W.T:

    - weight-only int8: quantize only W (per-output-row symmetric int8),
      keep x in full precision.
    - int8-dynamic: additionally quantize x per-row (per-sample,
      symmetric, recomputed on the fly from the batch) before the matmul.

    Returns (mse_weight_only, mse_dynamic).
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    y_fp = x @ W.T
    W_hat = _sym_quant_per_row(W)

    y_wo = x @ W_hat.T
    mse_wo = float(np.mean((y_fp - y_wo) ** 2))

    x_hat = _sym_quant_per_row(x)
    y_dyn = x_hat @ W_hat.T
    mse_dyn = float(np.mean((y_fp - y_dyn) ** 2))

    return mse_wo, mse_dyn
