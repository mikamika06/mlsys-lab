import math
import numpy as np


def stable_softmax_kernel(logits):
    x = np.asarray(logits, dtype=np.float64)
    n, d = x.shape
    result = np.zeros((n, d), dtype=np.float64)

    for r in range(n):
        max_val = x[r, 0]
        for c in range(1, d):
            val = x[r, c]
            if val > max_val:
                max_val = val

        row_sum = 0.0
        for c in range(d):
            exp_val = math.exp(x[r, c] - max_val)
            result[r, c] = exp_val
            row_sum += exp_val

        for c in range(d):
            result[r, c] = result[r, c] / row_sum

    trace = []
    for r in range(n):
        for c in range(d):
            trace.append((r * d + c) * 8)
    for r in range(n):
        for c in range(d):
            trace.append((r * d + c) * 8)

    return result, trace
