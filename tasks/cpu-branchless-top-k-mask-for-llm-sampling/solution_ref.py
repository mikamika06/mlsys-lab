import numpy as np


def branchless_topk_mask(logits: np.ndarray, k: int) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = -np.inf

    if k <= 0:
        return out

    vals = [float(logits[i]) for i in range(n)]

    for i in range(n):
        for j in range(0, n - i - 1):
            if vals[j] > vals[j + 1]:
                vals[j], vals[j + 1] = vals[j + 1], vals[j]

    tau = vals[n - k]

    for i in range(n):
        val = float(logits[i])
        if val >= tau:
            out[i] = val

    return out
