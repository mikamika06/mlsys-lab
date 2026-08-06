import numpy as np
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


def hqq_half_quadratic_step(W: np.ndarray, s: np.ndarray, z: np.ndarray,
                             W_q: np.ndarray, lp: float, beta: float,
                             qmin: int, qmax: int):
    """
    One HQQ half-quadratic-splitting iteration, per-row zero-point groups.

    1. raw = W/s + z            (per-element target before rounding)
    2. r   = W_q - raw           (residual against the GIVEN codes W_q)
    3. W_e = shrink_lp(r, beta, lp)   (generalized-Lp shrinkage/soft-threshold)
    4. z_new = mean_over_row(W_q - W_e - W/s)   (least-squares zero-point)
    5. W_q_new = clip(round(W/s + z_new), qmin, qmax)   (re-quantize)

    s, z are shape (d_out,) -- one scalar per row (group = row).
    Returns (W_q_new, z_new).
    """
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    W_q = np.asarray(W_q, dtype=np.float64)

    d_out, d_in = W.shape

    z_new_list = []
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
        z_new_list.append(z_new_i)

    z_new = np.asarray(z_new_list, dtype=np.float64)

    W_q_new_rows = []
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
        W_q_new_rows.append(row_W_q_new)

    W_q_new = np.asarray(W_q_new_rows, dtype=np.float64)

    return W_q_new, z_new
