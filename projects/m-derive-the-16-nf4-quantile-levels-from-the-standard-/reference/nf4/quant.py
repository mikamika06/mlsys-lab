import numpy as np
from scipy.stats import norm


def get_nf4_quantiles():
    n = 16
    q = np.arange(n + 1, dtype=np.float64) / n
    q[0] = 0.0
    q[-1] = 1.0
    q_all = norm.ppf(0.5 * (q[:-1] + q[1:]))
    q_all = q_all / q_all[-1]
    return q_all
