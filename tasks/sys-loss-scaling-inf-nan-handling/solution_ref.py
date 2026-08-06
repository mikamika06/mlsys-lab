import math
import numpy as np


def unscale_and_check(scaled_grads, scale):
    skip = False
    for g in scaled_grads:
        arr = np.asarray(g, dtype=np.float32)
        finite = True
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            val = float(arr[it.multi_index])
            if not math.isfinite(val):
                finite = False
                break
            it.iternext()
        if not finite:
            skip = True
            break

    unscaled = []
    for g in scaled_grads:
        arr = np.asarray(g, dtype=np.float32)
        res = np.empty_like(arr, dtype=np.float32)
        scale_f = np.float32(scale)
        it = np.nditer(arr, flags=["multi_index"], op_flags=["readonly"])
        while not it.finished:
            idx = it.multi_index
            res[idx] = np.float32(float(arr[idx]) / float(scale_f))
            it.iternext()
        unscaled.append(res)

    return bool(skip), unscaled
