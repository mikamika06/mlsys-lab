import math
import numpy as np


def quantile_keep_mask(w: np.ndarray, s: float):
    flat = np.asarray(w).reshape(-1)
    n = len(flat)
    mags = [float(abs(flat[i])) for i in range(n)]

    idx_float = s * (float(n) - 1.0)
    lower = math.floor(idx_float)
    upper = math.ceil(idx_float)
    weight = idx_float - float(lower)

    sorted_mags = sorted(mags)
    threshold = sorted_mags[lower] * (1.0 - weight) + sorted_mags[upper] * weight

    k = int(math.ceil((1.0 - s) * float(n)))

    order_items = sorted(((mags[i], i) for i in range(n)), key=lambda x: (-x[0], x[1]))
    order = [item[1] for item in order_items]

    mask_list = [False] * n
    for i in range(k):
        mask_list[order[i]] = True

    mask = np.array(mask_list, dtype=bool)

    return float(threshold), mask.reshape(np.asarray(w).shape)
