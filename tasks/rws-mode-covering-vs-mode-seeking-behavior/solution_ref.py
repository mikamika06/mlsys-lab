import math
import numpy as np

_EPS = 1e-12


def kd_divergence_family(p, students):
    """Forward-KL, reverse-KL, and JSD between a bimodal teacher and every
    member of a single-mode student family.

    `p` is a discrete probability distribution over a fixed grid (a bimodal
    teacher). `students` is a 2-D array where each row is a candidate
    single-mode student distribution over the same grid.

    For each candidate student `q = students[i]`:

        forward_kl[i] = KL(p || q) = sum_x p(x) * log(p(x) / q(x))
        reverse_kl[i] = KL(q || p) = sum_x q(x) * log(q(x) / p(x))
        m_i           = 0.5 * (p + q)
        jsd[i]        = 0.5 * KL(p || m_i) + 0.5 * KL(q || m_i)

    A small epsilon (1e-12) is added inside every log argument to avoid
    log(0).

    Parameters
    ----------
    p : np.ndarray, shape (G,)
    students : np.ndarray, shape (n_candidates, G)

    Returns
    -------
    dict with keys "forward_kl", "reverse_kl", "jsd". Each value is a tuple
    `(values, argmin)`:
        values : np.ndarray, float64, shape (n_candidates,)
        argmin : int, index of the candidate minimizing that divergence.
    """
    p = np.asarray(p, dtype=np.float64)
    Q = np.asarray(students, dtype=np.float64)

    n_candidates, G = Q.shape

    def kl_single(a, b):
        total = 0.0
        for x_idx in range(G):
            total += a[x_idx] * (math.log(a[x_idx] + _EPS) - math.log(b[x_idx] + _EPS))
        return total

    forward_list = []
    reverse_list = []
    jsd_list = []

    for i in range(n_candidates):
        q = Q[i]
        forward_list.append(kl_single(p, q))
        reverse_list.append(kl_single(q, p))
        m = 0.5 * (p + q)
        jsd_list.append(0.5 * kl_single(p, m) + 0.5 * kl_single(q, m))

    def process_vals(vals_list):
        vals = np.array(vals_list, dtype=np.float64)
        min_val = vals[0]
        min_idx = 0
        for idx in range(1, len(vals)):
            if vals[idx] < min_val:
                min_val = vals[idx]
                min_idx = idx
        return vals, int(min_idx)

    out = {}
    out["forward_kl"] = process_vals(forward_list)
    out["reverse_kl"] = process_vals(reverse_list)
    out["jsd"] = process_vals(jsd_list)
    return out
