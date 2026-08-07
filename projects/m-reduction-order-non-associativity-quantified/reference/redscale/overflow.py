import numpy as np


def global_overflow_skip(local_flags):
    arr = np.array(local_flags, dtype=bool)
    if arr.size == 0:
        return False
    global_flag = bool(np.any(arr))
    return global_flag
