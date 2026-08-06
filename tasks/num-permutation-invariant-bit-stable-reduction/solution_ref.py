import math
import numpy as np


def _key(x):
    bits = np.asarray(x, dtype=np.float64).view(np.uint64)
    if (bits >> np.uint64(63)) != 0:
        return bits ^ np.uint64(0xFFFFFFFFFFFFFFFF)
    else:
        return bits ^ np.uint64(0x8000000000000000)


def stable_sum(values):
    flat_list = []
    arr = np.asarray(values, dtype=np.float64)
    for v in arr.flat:
        flat_list.append(v)

    indexed = []
    for idx, v in enumerate(flat_list):
        k = _key(v)
        indexed.append((k, idx, v))

    n = len(indexed)
    for i in range(1, n):
        key_item = indexed[i]
        j = i - 1
        while j >= 0:
            if indexed[j][0] > key_item[0]:
                indexed[j + 1] = indexed[j]
                j -= 1
            else:
                break
        indexed[j + 1] = key_item

    ordered = [item[2] for item in indexed]

    first_nan = None
    for v in ordered:
        if math.isnan(v):
            first_nan = v
            break

    if first_nan is not None:
        return np.float64(first_nan)

    current = ordered
    while len(current) > 1:
        nxt = []
        i = 0
        while i + 1 < len(current):
            nxt.append(np.float64(current[i] + current[i + 1]))
            i += 2
        if i < len(current):
            nxt.append(current[i])
        current = nxt

    return np.float64(current[0])
