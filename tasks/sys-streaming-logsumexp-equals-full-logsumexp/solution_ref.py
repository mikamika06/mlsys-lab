import math
import numpy as np


def streaming_logsumexp(chunks):
    chunks = [np.asarray(c, dtype=np.float64) for c in chunks]
    N = chunks[0].shape[0]

    m = np.full(N, -float("inf"), dtype=np.float64)
    ell = np.zeros(N, dtype=np.float64)

    for c in chunks:
        nrows = c.shape[0]
        ncols = c.shape[1]
        for i in range(nrows):
            c_max = -float("inf")
            for j in range(ncols):
                val = c[i, j]
                if val > c_max:
                    c_max = val
            m_old = m[i]
            m_new = m_old if m_old > c_max else c_max
            term = 0.0
            for j in range(ncols):
                term += math.exp(c[i, j] - m_new)
            ell[i] = ell[i] * math.exp(m_old - m_new) + term
            m[i] = m_new

    res = np.empty(N, dtype=np.float64)
    for i in range(N):
        res[i] = m[i] + math.log(ell[i])
    return res
