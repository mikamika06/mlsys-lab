import numpy as np


def kahan_sum_fp16(x: np.ndarray) -> float:
    s = np.float32(0.0)
    c = np.float32(0.0)
    for value in x:
        y = np.float32(value) - c
        t = s + y
        c = (t - s) - y
        s = t
    return float(s)
