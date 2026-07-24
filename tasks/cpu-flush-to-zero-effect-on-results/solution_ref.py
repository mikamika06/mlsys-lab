import numpy as np


def dot_ftz_trace(a: np.ndarray, b: np.ndarray) -> tuple[float, list[int]]:
    def ftz(x):
        if 0 < abs(x) < np.finfo(np.float64).tiny:
            return 0.0
        return x

    acc = 0.0
    for i in range(len(a)):
        acc += ftz(float(a[i]) * float(b[i]))

    trace = []
    for i in range(len(a)):
        trace.append(i * 8)
        trace.append(4096 + i * 8)

    return acc, trace
