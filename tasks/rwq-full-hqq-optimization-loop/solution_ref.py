import numpy as np


def _shrink_op(x: np.ndarray, beta: float, p: float) -> np.ndarray:
    """Proximal operator of beta * ||.||_p^p (elementwise soft/half-quadratic shrink)."""
    res = np.empty_like(x, dtype=np.float64)
    flat_x = x.ravel()
    flat_res = res.ravel()
    n = flat_x.size
    inv_beta = 1.0 / beta
    if p == 1.0:
        for i in range(n):
            val = flat_x[i]
            abs_val = abs(val)
            sign_val = -1.0 if val < 0 else (1.0 if val > 0 else 0.0)
            flat_res[i] = sign_val * max(abs_val - inv_beta, 0.0)
    else:
        p_minus_1 = p - 1.0
        for i in range(n):
            val = flat_x[i]
            abs_val = abs(val)
            sign_val = -1.0 if val < 0 else (1.0 if val > 0 else 0.0)
            term = abs_val - inv_beta * (abs_val ** p_minus_1)
            flat_res[i] = sign_val * max(term, 0.0)
    return res


def hqq_optimize(W, scale, zero0, qmin, qmax, lp_norm, beta0, kappa, iters):
    """
    HQQ-style zero-point optimization: scale is held fixed; the zero-point z
    is refined for `iters` half-quadratic passes, then W is quantized one
    final time with the converged z. Returns (W_q, z, W_dequant).
    """
    W = np.asarray(W, dtype=np.float64)
    zero = float(zero0)
    beta = float(beta0)

    n = W.size
    flat_W = W.ravel()

    for _ in range(iters):
        W_q_arr = np.empty_like(W, dtype=np.float64)
        flat_W_q = W_q_arr.ravel()
        for i in range(n):
            val = round(flat_W[i] * scale + zero)
            flat_W_q[i] = min(max(val, qmin), qmax)

        W_r_arr = np.empty_like(W, dtype=np.float64)
        flat_W_r = W_r_arr.ravel()
        for i in range(n):
            flat_W_r[i] = (flat_W_q[i] - zero) / scale

        diff_arr = np.empty_like(W, dtype=np.float64)
        flat_diff = diff_arr.ravel()
        for i in range(n):
            flat_diff[i] = flat_W[i] - flat_W_r[i]

        W_e_arr = _shrink_op(diff_arr, beta, lp_norm)
        flat_W_e = W_e_arr.ravel()

        sum_val = 0.0
        for i in range(n):
            term = flat_W_q[i] - (flat_W[i] - flat_W_e[i]) * scale
            sum_val += term
        zero = float(sum_val / n)

        beta *= kappa

    W_q_arr = np.empty_like(W, dtype=np.float64)
    flat_W_q = W_q_arr.ravel()
    for i in range(n):
        val = round(flat_W[i] * scale + zero)
        flat_W_q[i] = min(max(val, qmin), qmax)

    W_dq_arr = np.empty_like(W, dtype=np.float64)
    flat_W_dq = W_dq_arr.ravel()
    for i in range(n):
        flat_W_dq[i] = (flat_W_q[i] - zero) / scale

    return W_q_arr, zero, W_dq_arr
