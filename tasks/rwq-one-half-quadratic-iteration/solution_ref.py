import math

EPS = 1e-8


def _shrink_lp(x, beta, p):
    ax = abs(x)
    if p == 1.0:
        mag = 1.0 / beta
    else:
        mag = (1.0 / beta) * math.pow(ax + EPS, p - 1.0)
    sig = 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)
    res = ax - mag
    max_res = res if res > 0.0 else 0.0
    return sig * max_res


def hqq_half_quadratic_step(W: list[list[float]], s: list[float], z: list[float],
                             W_q: list[list[float]], lp: float, beta: float,
                             qmin: int, qmax: int) -> tuple[list[list[float]], list[float]]:
    """
    One HQQ half-quadratic-splitting iteration, per-row zero-point groups.

    1. raw = W/s + z            (per-element target before rounding)
    2. r   = W_q - raw           (residual against the GIVEN codes W_q)
    3. W_e = shrink_lp(r, beta, lp)   (generalized-Lp shrinkage/soft-threshold)
    4. z_new = mean_over_row(W_q - W_e - W/s)   (least-squares zero-point)
    5. W_q_new = clip(round(W/s + z_new), qmin, qmax)   (re-quantize)

    s, z are length d_out -- one scalar per row (group = row).
    Returns (W_q_new, z_new).
    """
    d_out = len(W)
    d_in = len(W[0])

    z_new = []
    for i in range(d_out):
        si = s[i]
        zi = z[i]
        row_W = W[i]
        row_W_q = W_q[i]

        row_term_sum = 0.0
        for j in range(d_in):
            wij = row_W[j]
            wq_ij = row_W_q[j]
            raw_ij = wij / si + zi
            r_ij = wq_ij - raw_ij
            we_ij = _shrink_lp(r_ij, beta, lp)

            term_ij = wq_ij - we_ij - wij / si
            row_term_sum += term_ij

        z_new_i = row_term_sum / float(d_in)
        z_new.append(z_new_i)

    W_q_new = []
    for i in range(d_out):
        si = s[i]
        z_new_i = z_new[i]
        row_W = W[i]
        row_W_q_new = []
        for j in range(d_in):
            wij = row_W[j]
            val = wij / si + z_new_i
            rnd = round(val)
            clipped = float(qmin) if rnd < qmin else (float(qmax) if rnd > qmax else float(rnd))
            row_W_q_new.append(clipped)
        W_q_new.append(row_W_q_new)

    return W_q_new, z_new
