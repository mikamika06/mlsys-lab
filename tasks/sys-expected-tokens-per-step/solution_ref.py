import numpy as np


def expected_tokens_per_step(accept_probs) -> float:
    p = np.asarray(accept_probs, dtype=np.float64)
    K = p.shape[0]
    if K == 0:
        return 1.0
    total = 0.0
    prod = 1.0
    for i in range(K):
        prod *= p[i]
        total += prod
    return float(1.0 + total)
