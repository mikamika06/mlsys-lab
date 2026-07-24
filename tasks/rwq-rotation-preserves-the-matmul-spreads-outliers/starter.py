import numpy as np


def rotate_and_quantize_matmul(X, W):
    """QuaRot-style rotation invariance and its quantization payoff.

    X: (n, d), d a power of two. W: (d, m).

    Build the normalized Sylvester-Hadamard matrix H of size d (H^T H = I).
    Let X' = X @ H.T and W' = H @ W; then X' @ W' == X @ W exactly.

    Symmetric per-tensor int4 round-trip a tensor t as: qmax = 7,
    scale = max(|t|) / qmax (or 1.0 if all-zero), code = clip(round(t /
    scale), -qmax, qmax), dequant = code * scale.

    Returns (out_rotated, mse_unrotated, mse_rotated):
      out_rotated   -- X' @ W', float64, shape (n, m).
      mse_unrotated -- float, mean((X@W - Xq@Wq)**2) using int4 round-tripped
                       (unrotated) X, W.
      mse_rotated   -- float, mean((X@W - Xq'@Wq')**2) using int4
                       round-tripped (rotated) X', W'.
    """
    raise NotImplementedError('your code here')
