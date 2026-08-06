import math
import numpy as np

_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def snap_to_e2m1(x):
    """Snap each element of x to the nearest signed E2M1 FP4 value."""
    x = np.asarray(x, dtype=np.float64)
    out = []
    for val in x.ravel():
        abs_val = math.fabs(val)
        min_diff = float("inf")
        best_mag = _MAG[0]
        for mag in _MAG:
            diff = math.fabs(abs_val - mag)
            if diff < min_diff:
                min_diff = diff
                best_mag = mag
        out.append(math.copysign(best_mag, val))
    return np.array(out, dtype=np.float64).reshape(x.shape)
