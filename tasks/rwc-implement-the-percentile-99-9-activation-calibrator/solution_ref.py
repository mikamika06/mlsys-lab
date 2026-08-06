import math
import numpy as np


def percentile_amax(calibration_batches):
    flat_list = []
    for batch in calibration_batches:
        arr = np.asarray(batch)
        flat = arr.reshape(-1)
        for i in range(flat.shape[0]):
            val = flat[i]
            if val < 0.0:
                abs_val = -val
            else:
                abs_val = val
            flat_list.append(abs_val)

    n = len(flat_list)
    if n == 0:
        return 0.0

    for i in range(n - 1):
        for j in range(n - 1 - i):
            if flat_list[j] > flat_list[j + 1]:
                temp = flat_list[j]
                flat_list[j] = flat_list[j + 1]
                flat_list[j + 1] = temp

    k = (99.9 / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        ans = flat_list[int(k)]
    else:
        d0 = flat_list[int(f)] * (c - k)
        d1 = flat_list[int(c)] * (k - f)
        ans = d0 + d1

    return float(ans)
