import numpy as np


def _accepted_tokens(alpha, k):
    max_k = 0
    for val in k:
        if val > max_k:
            max_k = int(val)

    total = np.zeros_like(alpha)
    term = np.ones_like(alpha)
    
    for i in range(max_k + 1):
        for idx in range(len(alpha)):
            if i <= k[idx]:
                total[idx] = total[idx] + term[idx]
            term[idx] = term[idx] * alpha[idx]
            
    return total


def break_even_alpha(configs: np.ndarray) -> np.ndarray:
    configs = np.asarray(configs, dtype=np.float64)
    c = configs[:, 0]
    k = configs[:, 1].astype(np.int64)

    target = np.zeros_like(c)
    for i in range(len(c)):
        target[i] = 1.0 + k[i] * c[i]

    lo = np.zeros_like(c)
    hi = np.ones_like(c)

    for _ in range(200):
        mid = np.zeros_like(c)
        for i in range(len(c)):
            mid[i] = (lo[i] + hi[i]) / 2.0

        val = _accepted_tokens(mid, k)

        for i in range(len(c)):
            if val[i] < target[i]:
                lo[i] = mid[i]
            else:
                hi[i] = mid[i]

    result = np.zeros_like(c)
    for i in range(len(c)):
        result[i] = (lo[i] + hi[i]) / 2.0

    return result
