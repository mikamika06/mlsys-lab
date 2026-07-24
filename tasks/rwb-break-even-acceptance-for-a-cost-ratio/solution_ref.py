import numpy as np


def _accepted_tokens(alpha, k):
    max_k = int(np.max(k))
    total = np.zeros_like(alpha)
    term = np.ones_like(alpha)
    for i in range(max_k + 1):
        active = i <= k
        total = total + np.where(active, term, 0.0)
        term = term * alpha
    return total


def break_even_alpha(configs: np.ndarray) -> np.ndarray:
    configs = np.asarray(configs, dtype=np.float64)
    c = configs[:, 0]
    k = configs[:, 1].astype(np.int64)

    target = 1.0 + k * c

    lo = np.zeros_like(c)
    hi = np.ones_like(c)

    for _ in range(200):
        mid = (lo + hi) / 2.0
        val = _accepted_tokens(mid, k)
        go_up = val < target
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)

    return (lo + hi) / 2.0
