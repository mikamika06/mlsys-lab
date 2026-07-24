import numpy as np


def ring_merge(partials):
    ms = [np.asarray(p[0], dtype=np.float64) for p in partials]
    ls = [np.asarray(p[1], dtype=np.float64) for p in partials]
    accs = [np.asarray(p[2], dtype=np.float64) for p in partials]

    m_new = np.maximum.reduce(ms)
    total_l = np.zeros_like(m_new)
    total_a = np.zeros_like(accs[0], dtype=np.float64)

    for m, l, a in zip(ms, ls, accs):
        scale = np.exp(m - m_new)
        total_l += l * scale
        total_a += a * scale[:, None]

    return total_a / total_l[:, None]
