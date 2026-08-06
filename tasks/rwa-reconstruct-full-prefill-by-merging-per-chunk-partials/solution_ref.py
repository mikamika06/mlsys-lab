import math
import numpy as np


def merge_chunk_partials(ms, ls, os):
    ms = np.asarray(ms, dtype=np.float64)
    ls = np.asarray(ls, dtype=np.float64)
    os = np.asarray(os, dtype=np.float64)

    n_chunks = ms.shape[0]
    
    M = ms[0]
    for i in range(1, n_chunks):
        if ms[i] > M:
            M = ms[i]

    scale = np.empty(n_chunks, dtype=np.float64)
    for i in range(n_chunks):
        scale[i] = math.exp(ms[i] - M)

    L = 0.0
    for i in range(n_chunks):
        L += ls[i] * scale[i]

    d = os.shape[1]
    O = np.zeros(d, dtype=np.float64)
    for j in range(d):
        acc = 0.0
        for i in range(n_chunks):
            acc += os[i, j] * scale[i]
        O[j] = acc

    return O / L
