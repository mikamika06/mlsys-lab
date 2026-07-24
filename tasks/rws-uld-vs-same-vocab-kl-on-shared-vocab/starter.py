import numpy as np


def uld_and_kl_along_sweep(p_teacher: np.ndarray, p_students: np.ndarray):
    """Compute ULD and same-vocab KL between a teacher distribution and a
    sweep of student distributions sharing the same vocabulary/index order.

    p_teacher: shape (V,), a probability distribution (sums to 1, all > 0).
    p_students: shape (n, V), n probability distributions (each sums to 1,
      all > 0); one row is exactly equal to p_teacher.

    For every row p_s of p_students, compute:
      uld  = sum(|sort(p_teacher) - sort(p_s)|)          (sorted L1 / Wasserstein-1)
      kl   = sum(p_teacher * log(p_teacher / p_s))       (same-index KL(p_t || p_s))

    Returns (uld_values, kl_values), each shape (n,).
    """
    raise NotImplementedError('your code here')
