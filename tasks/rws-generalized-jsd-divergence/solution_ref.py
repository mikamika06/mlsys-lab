import numpy as np

def generalized_jsd(p: np.ndarray, q: np.ndarray, beta: float) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    beta = float(beta)
    m = beta * p + (1 - beta) * q
    eps = 1e-12
    kl_p_m = np.sum(np.where(p > 0, p * np.log(p / (m + eps)), 0.0))
    kl_q_m = np.sum(np.where(q > 0, q * np.log(q / (m + eps)), 0.0))
    return beta * kl_p_m + (1 - beta) * kl_q_m
