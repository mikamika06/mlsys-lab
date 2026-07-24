import numpy as np


def _hadamard(n):
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _quantize_int4(x):
    """Symmetric per-tensor absmax int4 round-trip (quant then immediate dequant)."""
    x = np.asarray(x, dtype=np.float64)
    qmax = 2 ** (4 - 1) - 1  # 7
    scale = float(np.max(np.abs(x)))
    scale = scale / qmax if scale > 0 else 1.0
    code = np.clip(np.round(x / scale), -qmax, qmax)
    return code * scale


def rotate_and_quantize_matmul(X, W):
    """QuaRot-style rotation invariance and its quantization payoff.

    An orthogonal rotation H (a normalized Sylvester-Hadamard matrix) folded
    into both operands of a matmul leaves the product exactly unchanged,
    because H^T H = I:

        X' = X @ H^T,   W' = H @ W   =>   X' @ W' = X @ (H^T H) @ W = X @ W.

    But the rotation mixes every output channel of X into every rotated
    channel, so a handful of huge outlier activation channels get spread
    across all d channels instead of dominating one. A per-tensor int4
    quantizer -- whose single scale is set by the single largest-magnitude
    element -- benefits enormously: rotate first, quantize second.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Activations. `d` must be a power of two.
    W : np.ndarray, shape (d, m)
        Weights.

    Returns
    -------
    out_rotated : np.ndarray, float64, shape (n, m)
        `X' @ W'`, the matmul computed entirely in the rotated basis. Equal
        to `X @ W` up to floating point error.
    mse_unrotated : float
        Mean squared error between `X @ W` and `Xq @ Wq`, where `Xq`, `Wq`
        are symmetric per-tensor int4 round-tripped versions of the
        *unrotated* `X`, `W`.
    mse_rotated : float
        Mean squared error between `X @ W` and `Xq' @ Wq'`, where `Xq'`,
        `Wq'` are symmetric per-tensor int4 round-tripped versions of the
        *rotated* `X'`, `W'`.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    d = X.shape[1]
    H = _hadamard(d)

    ref = X @ W

    Xr = X @ H.T
    Wr = H @ W
    out_rotated = Xr @ Wr

    Xq = _quantize_int4(X)
    Wq = _quantize_int4(W)
    mse_unrotated = float(np.mean((ref - Xq @ Wq) ** 2))

    Xrq = _quantize_int4(Xr)
    Wrq = _quantize_int4(Wr)
    mse_rotated = float(np.mean((ref - Xrq @ Wrq) ** 2))

    return out_rotated, mse_unrotated, mse_rotated
