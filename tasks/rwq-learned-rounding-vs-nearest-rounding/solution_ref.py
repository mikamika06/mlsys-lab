import itertools
import math

import numpy as np


def _row_scale(w, qmax):
    a = 0.0
    for val in w:
        v_abs = val if val >= 0.0 else -val
        if v_abs > a:
            a = v_abs
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
    combos = list(itertools.product([0, 1], repeat=d_in))

    total_learned = 0.0
    total_rtn = 0.0
    n_cal = X.shape[0]

    for i in range(d_out):
        w = W[i]
        s = _row_scale(w, qmax)
        
        y = [0.0] * n_cal
        for k in range(n_cal):
            s_val = 0.0
            X_k = X[k]
            for j in range(d_in):
                s_val += w[j] * X_k[j]
            y[k] = s_val

        cand_f = [0.0] * d_in
        cand_c = [0.0] * d_in
        for j in range(d_in):
            val = w[j] / s
            f_val = math.floor(val)
            if f_val < -qmax:
                f_val = -qmax
            elif f_val > qmax:
                f_val = qmax
            cand_f[j] = f_val

            c_val = math.ceil(val)
            if c_val < -qmax:
                c_val = -qmax
            elif c_val > qmax:
                c_val = qmax
            cand_c[j] = c_val

        min_sse = None
        for combo in combos:
            sse_combo = 0.0
            for k in range(n_cal):
                X_k = X[k]
                yhat_k = 0.0
                for j in range(d_in):
                    choice = combo[j]
                    chosen_val = cand_f[j] if choice == 0 else cand_c[j]
                    v_j = chosen_val * s
                    yhat_k += v_j * X_k[j]
                diff = yhat_k - y[k]
                sse_combo += diff * diff
            if min_sse is None or sse_combo < min_sse:
                min_sse = sse_combo
        total_learned += float(min_sse)

        v_rtn = [0.0] * d_in
        for j in range(d_in):
            val = w[j] / s
            rtn_val = round(val)
            if rtn_val < -qmax:
                rtn_val = -qmax
            elif rtn_val > qmax:
                rtn_val = qmax
            v_rtn[j] = rtn_val * s

        sse_rtn = 0.0
        for k in range(n_cal):
            X_k = X[k]
            yhat_rtn_k = 0.0
            for j in range(d_in):
                yhat_rtn_k += v_rtn[j] * X_k[j]
            diff = yhat_rtn_k - y[k]
            sse_rtn += diff * diff
        total_rtn += float(sse_rtn)

    n = d_out * n_cal
    return total_learned / n, total_rtn / n
