import numpy as np
import math

def generalized_jsd(p: np.ndarray, q: np.ndarray, beta: float) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    beta = float(beta)
    eps = 1e-12
    kl_p_m = 0.0
    kl_q_m = 0.0
    n = len(p)
    for i in range(n):
        pi = p[i]
        qi = q[i]
        mi = beta * pi + (1 - beta) * qi
        if pi > 0:
            kl_p_m += pi * math.log(pi / (mi + eps))
        if qi > 0:
            kl_q_m += qi * math.log(qi / (mi + eps))
    return beta * kl_p_m + (1 - beta) * kl_q_m
