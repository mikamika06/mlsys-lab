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

    def kl(a, b):
        return np.sum(a * (np.log(a + _EPS) - np.log(b + _EPS)), axis=-1)

    forward = kl(p[None, :], Q)
    reverse = kl(Q, p[None, :])

    m = 0.5 * (p[None, :] + Q)
    jsd = 0.5 * kl(p[None, :], m) + 0.5 * kl(Q, m)

    out = {}
    for name, vals in (("forward_kl", forward), ("reverse_kl", reverse), ("jsd", jsd)):
        vals = np.asarray(vals, dtype=np.float64)
        out[name] = (vals, int(np.argmin(vals)))
    return out
