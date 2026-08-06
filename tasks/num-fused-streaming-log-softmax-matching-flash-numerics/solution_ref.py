import math
import numpy as np


def streaming_log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = -float("inf")
    for value in x:
        if value > m:
            m = value

    shifted = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        shifted[i] = x[i] - m

    total = 0.0
    for value in shifted:
        total += math.exp(value)

    log_total = math.log(total)

    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(shifted)):
        out[i] = shifted[i] - log_total

    return out
