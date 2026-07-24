import numpy as np


def streaming_logsumexp(chunks):
    chunks = [np.asarray(c, dtype=np.float64) for c in chunks]
    N = chunks[0].shape[0]

    m = np.full(N, -np.inf)
    ell = np.zeros(N)

    for c in chunks:
        c_max = np.max(c, axis=1)
        m_new = np.maximum(m, c_max)
        ell = ell * np.exp(m - m_new) + np.sum(np.exp(c - m_new[:, None]), axis=1)
        m = m_new

    return m + np.log(ell)
