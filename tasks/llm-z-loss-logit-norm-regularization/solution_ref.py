import math
import numpy as np


def z_loss(logits: np.ndarray, targets: np.ndarray, lambda_: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    N, C = logits.shape
    out = []
    for i in range(N):
        m = logits[i, 0]
        for j in range(1, C):
            val = logits[i, j]
            if val > m:
                m = val

        s = 0.0
        for j in range(C):
            s += math.exp(logits[i, j] - m)

        lse = m + math.log(s)
        ce = -logits[i, targets[i]] + lse
        out.append(ce + lambda_ * (lse ** 2))

    return np.array(out, dtype=np.float64)
