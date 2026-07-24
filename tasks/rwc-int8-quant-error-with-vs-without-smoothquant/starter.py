import numpy as np


def smoothquant_w8a8_comparison(X: np.ndarray, W: np.ndarray, alpha: float) -> dict:
    """Compare per-tensor symmetric INT8 W8A8 (weight + activation)
    quantization error for a linear layer Y = X @ W^T, with and without
    SmoothQuant's activation-outlier migration.

    X : (n, d_in) activations (outlier-heavy input channels).
    W : (d_out, d_in) weight matrix.
    alpha : float in (0, 1), SmoothQuant's migration strength.

    SmoothQuant computes a PER-INPUT-CHANNEL scale
        s_j = max_i(|X[i,j]|)^alpha / max_o(|W[o,j]|)^(1-alpha)
    and rescales X_smooth[:, j] = X[:, j] / s_j, W_smooth[:, j] = W[:, j] * s_j
    -- mathematically X_smooth @ W_smooth^T == X @ W^T exactly (no
    approximation), it only redistributes dynamic range from activations
    (hard to quantize, outlier-heavy) onto weights (easy to quantize,
    already well-behaved) BEFORE both are cast to INT8.

    Returns a dict:
      "error_raw"        : Frobenius relative error of the raw (no
                            smoothing) W8A8 output vs the exact FP output.
      "error_smoothed"    : same, with SmoothQuant migration applied
                            first.
      "improvement_ratio" : error_smoothed / error_raw (< 1 means
                            smoothing helped).
    """
    raise NotImplementedError('your code here')
