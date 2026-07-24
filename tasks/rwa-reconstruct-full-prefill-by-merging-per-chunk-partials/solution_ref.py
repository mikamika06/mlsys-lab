import numpy as np


def merge_chunk_partials(ms, ls, os):
    ms = np.asarray(ms, dtype=np.float64)
    ls = np.asarray(ls, dtype=np.float64)
    os = np.asarray(os, dtype=np.float64)

    M = np.max(ms)
    scale = np.exp(ms - M)
    L = np.sum(ls * scale)
    O = np.sum(os * scale[:, None], axis=0)

    return O / L
