import numpy as np

def blockwise_absmax(w, block_size=64):
    """Return per-block maximum absolute value of w."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    pad_len = (-n) % block_size
    if pad_len:
        w = np.concatenate([w, np.zeros(pad_len)])
    num_blocks = len(w) // block_size
    res = []
    for i in range(num_blocks):
        m = 0.0
        start = i * block_size
        end = start + block_size
        for j in range(start, end):
            val = abs(w[j])
            if val > m:
                m = val
        res.append(m)
    return np.asarray(res, dtype=np.float64)
