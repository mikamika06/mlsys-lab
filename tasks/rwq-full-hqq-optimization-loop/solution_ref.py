def _shrink_op(x: list[float], beta: float, p: float) -> list[float]:
    """Proximal operator of beta * ||.||_p^p (elementwise soft/half-quadratic shrink)."""
    res = []
    inv_beta = 1.0 / beta
    if p == 1.0:
        for val in x:
            abs_val = abs(val)
            sign_val = -1.0 if val < 0 else (1.0 if val > 0 else 0.0)
            res.append(sign_val * max(abs_val - inv_beta, 0.0))
    else:
        p_minus_1 = p - 1.0
        for val in x:
            abs_val = abs(val)
            sign_val = -1.0 if val < 0 else (1.0 if val > 0 else 0.0)
            term = abs_val - inv_beta * (abs_val ** p_minus_1)
            res.append(sign_val * max(term, 0.0))
    return res


def hqq_optimize(W: list[float], scale: float, zero0: float, qmin: int, qmax: int, lp_norm: float, beta0: float, kappa: float, iters: int) -> tuple[list[int], float, list[float]]:
    """
    HQQ-style zero-point optimization: scale is held fixed; the zero-point z
    is refined for `iters` half-quadratic passes, then W is quantized one
    final time with the converged z. Returns (W_q, z, W_dequant).
    """
    zero = float(zero0)
    beta = float(beta0)

    n = len(W)
    current_W = list(W)

    for _ in range(iters):
        W_q_list = []
        for i in range(n):
            val = round(current_W[i] * scale + zero)
            W_q_list.append(int(min(max(val, qmin), qmax)))

        W_r_list = []
        for i in range(n):
            W_r_list.append((W_q_list[i] - zero) / scale)

        diff_list = []
        for i in range(n):
            diff_list.append(current_W[i] - W_r_list[i])

        W_e_list = _shrink_op(diff_list, beta, lp_norm)

        sum_val = 0.0
        for i in range(n):
            term = W_q_list[i] - (current_W[i] - W_e_list[i]) * scale
            sum_val += term
        zero = float(sum_val / n)

        beta *= kappa

    W_q_final = []
    for i in range(n):
        val = round(current_W[i] * scale + zero)
        W_q_final.append(int(min(max(val, qmin), qmax)))

    W_dq_list = []
    for i in range(n):
        W_dq_list.append((W_q_final[i] - zero) / scale)

    return W_q_final, zero, W_dq_list
