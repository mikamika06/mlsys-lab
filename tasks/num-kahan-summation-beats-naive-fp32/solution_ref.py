import math
import numpy as np


def compensated_sum(arr: np.ndarray) -> float:
    """Kahan-Neumaier compensated summation, accumulating entirely in
    float32. Returns a Python float."""
    s = np.float32(0.0)
    c = np.float32(0.0)
    for x in arr:
        x = np.float32(x)
        t = np.float32(s + x)
        if math.fabs(s) >= math.fabs(x):
            c = np.float32(c + np.float32(np.float32(s - t) + x))
        else:
            c = np.float32(c + np.float32(np.float32(x - t) + s))
        s = t
    return float(np.float32(s + c))
