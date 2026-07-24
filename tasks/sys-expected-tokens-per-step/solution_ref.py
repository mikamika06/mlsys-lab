import numpy as np


def expected_tokens_per_step(accept_probs) -> float:
    p = np.asarray(accept_probs, dtype=np.float64)
    if p.shape[0] == 0:
        return 1.0
    return float(1.0 + np.sum(np.cumprod(p)))
