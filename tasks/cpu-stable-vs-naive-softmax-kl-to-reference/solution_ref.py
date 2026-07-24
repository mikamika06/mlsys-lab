import numpy as np


def stable_softmax_kernel(logits):
    x = np.asarray(logits, dtype=np.float64)
    maximum = np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(x - maximum)
    result = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    n, d = x.shape
    trace = []
    for r in range(n):
        for c in range(d):
            trace.append((r * d + c) * 8)
    for r in range(n):
        for c in range(d):
            trace.append((r * d + c) * 8)

    return result, trace
