import numpy as np


def pack_int4(values):
    values = np.asarray(values, dtype=np.int64).reshape(-1)
    out = np.zeros((values.size + 1) // 2, dtype=np.uint8)
    for i in range(0, values.size, 2):
        low = values[i] & 15
        high = values[i + 1] & 15 if i + 1 < values.size else 0
        out[i // 2] = np.uint8(low | (high << 4))
    return out
