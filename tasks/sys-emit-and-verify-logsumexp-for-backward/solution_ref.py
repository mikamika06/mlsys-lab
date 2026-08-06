import math
import numpy as np


def emit_lse(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    rows, cols = S.shape
    out = []
    for i in range(rows):
        m = S[i, 0]
        for j in range(1, cols):
            v = S[i, j]
            if v > m:
                m = v
        total = 0.0
        for j in range(cols):
            total += math.exp(S[i, j] - m)
        out.append(m + math.log(total))
    return np.array(out, dtype=np.float64)
