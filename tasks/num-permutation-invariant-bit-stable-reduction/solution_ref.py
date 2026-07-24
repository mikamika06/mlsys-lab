import numpy as np


def _key(x):
    bits = np.asarray(x, dtype=np.float64).view(np.uint64)
    return np.where((bits >> np.uint64(63)) != 0,
                    bits ^ np.uint64(0xFFFFFFFFFFFFFFFF),
                    bits ^ np.uint64(0x8000000000000000))


def stable_sum(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    ordered = values[np.argsort(_key(values), kind="stable")]

    nan_positions = np.flatnonzero(np.isnan(ordered))
    if len(nan_positions):
        return np.float64(ordered[nan_positions[0]])

    current = list(ordered)
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
