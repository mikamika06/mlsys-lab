import numpy as np

def blockwise_absmax(w, block_size=64):
    """Return per-block maximum absolute value of w."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    pad_len = (-n) % block_size
    if pad_len:
        w = np.concatenate([w, np.zeros(pad_len)])
    return np.max(np.abs(w.reshape(-1, block_size)), axis=1)
