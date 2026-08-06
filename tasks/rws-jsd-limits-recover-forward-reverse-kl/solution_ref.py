import math
import numpy as np


def jsd_beta(p: np.ndarray, q: np.ndarray, beta: float) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    kl_q = 0.0
    kl_p = 0.0
    for i in range(len(p)):
        pi = p[i]
        qi = q[i]
        mi = beta * qi + (1.0 - beta) * pi
        kl_q += qi * math.log(qi / mi)
        kl_p += pi * math.log(pi / mi)
    return float(
        (beta * kl_q + (1.0 - beta) * kl_p)
        / (beta * (1.0 - beta))
    )
