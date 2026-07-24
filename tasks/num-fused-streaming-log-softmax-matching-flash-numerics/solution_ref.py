import numpy as np


def streaming_log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x)

    total = 0.0
    for value in x:
        total += np.exp(value - m)

    return (x - m) - np.log(total)
