import numpy as np

def compensated_sum(arr: np.ndarray) -> np.float32:
    s = np.float32(0.0)
    c = np.float32(0.0)
    for x in arr:
        t = np.float32(s + x)
        if np.abs(s) >= np.abs(x):
            c = np.float32(c + np.float32(np.float32(s - t) + x))
        else:
            c = np.float32(c + np.float32(np.float32(x - t) + s))
        s = t
    return np.float32(s + c)
