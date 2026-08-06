import numpy as np


def compensated_sum(arr: np.ndarray) -> np.float32:
    s = np.float32(0.0)
    c = np.float32(0.0)
    for x in arr:
        t = np.float32(s + x)
        abs_s = s if s >= 0 else -s
        abs_x = x if x >= 0 else -x
        if abs_s >= abs_x:
            c = np.float32(c + np.float32(np.float32(s - t) + x))
        else:
            c = np.float32(c + np.float32(np.float32(x - t) + s))
        s = t
    return np.float32(s + c)
