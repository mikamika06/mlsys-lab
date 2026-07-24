import itertools

import numpy as np


def _row_scale(w, qmax):
    a = float(np.max(np.abs(w)))
    return a / qmax if a > 0 else 1.0


def rounding_output_mse(W: np.ndarray, X: np.ndarray, nbits: int):
    """
    For each output row (weight vector) independently:

    - Compute a symmetric per-row scale s = max(|w|) / qmax, qmax = 2^(b-1)-1.
    - "Learned" rounding: brute-force, over all 2^d_in choices of rounding
      each element of w/s DOWN or UP (then clipping to [-qmax, qmax]), pick
      the one minimizing the row's output squared error
      ||X @ (code*s) - X @ w||^2.
    - "RTN" rounding: nearest-integer rounding of w/s (clipped the same way).

    Returns (mse_learned, mse_rtn): each the mean squared output error,
    averaged over all rows and all calibration samples in X.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    d_out, d_in = W.shape
    qmax = (1 << (nbits - 1)) - 1
    combos = np.array(list(itertools.product([0, 1], repeat=d_in)))

    total_learned = 0.0
    total_rtn = 0.0
    for i in range(d_out):
        w = W[i]
        s = _row_scale(w, qmax)
        y = w @ X.T

        f = np.clip(np.floor(w / s), -qmax, qmax)
        c = np.clip(np.ceil(w / s), -qmax, qmax)
        cand = np.stack([f, c], axis=1)
        chosen = cand[np.arange(d_in)[None, :], combos]
        v = chosen * s
        Yhat = v @ X.T
        sse = np.sum((Yhat - y[None, :]) ** 2, axis=1)
        total_learned += float(np.min(sse))

        code_rtn = np.clip(np.round(w / s), -qmax, qmax)
        v_rtn = code_rtn * s
        total_rtn += float(np.sum((v_rtn @ X.T - y) ** 2))

    n = d_out * X.shape[0]
    return total_learned / n, total_rtn / n
