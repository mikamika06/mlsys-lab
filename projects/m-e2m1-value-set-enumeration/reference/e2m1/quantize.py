import numpy as np
from e2m1.enumeration import enumerate_e2m1

def quantize_e2m1(x):
    arr = np.asarray(x, dtype=np.float32)
    table = enumerate_e2m1()
    vals = np.array([t["value"] for t in table], dtype=np.float32)
    shape = arr.shape
    flat = arr.flatten()
    out = np.empty_like(flat)
    for i, val in enumerate(flat):
        diffs = np.abs(vals - val)
        min_idx = np.argmin(diffs)
        out[i] = vals[min_idx]
    return out.reshape(shape)
