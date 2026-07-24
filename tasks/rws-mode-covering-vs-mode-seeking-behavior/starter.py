import numpy as np


def kd_divergence_family(p, students):
    """Forward-KL, reverse-KL, and JSD between a bimodal teacher and every
    member of a single-mode student family.

    p: 1-D array, shape (G,), a discrete probability distribution (bimodal
    teacher) over a fixed grid. students: 2-D array, shape
    (n_candidates, G), each row a candidate single-mode student
    distribution over the same grid.

    For each candidate q = students[i], with eps = 1e-12 inside every log:
        forward_kl[i] = KL(p || q) = sum_x p(x) * log(p(x) / q(x))
        reverse_kl[i] = KL(q || p) = sum_x q(x) * log(q(x) / p(x))
        m_i           = 0.5 * (p + q)
        jsd[i]        = 0.5 * KL(p || m_i) + 0.5 * KL(q || m_i)

    Returns a dict with keys "forward_kl", "reverse_kl", "jsd", each
    mapping to a tuple (values, argmin):
        values -- float64 array, shape (n_candidates,)
        argmin -- int, index of the candidate minimizing that divergence.
    """
    raise NotImplementedError('your code here')
