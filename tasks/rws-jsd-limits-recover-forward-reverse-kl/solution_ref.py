import numpy as np


def jsd_beta(p: np.ndarray, q: np.ndarray, beta: float) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = beta * q + (1.0 - beta) * p
    kl_q = np.sum(q * np.log(q / m))
    kl_p = np.sum(p * np.log(p / m))
    return float(
        (beta * kl_q + (1.0 - beta) * kl_p)
        / (beta * (1.0 - beta))
    )
