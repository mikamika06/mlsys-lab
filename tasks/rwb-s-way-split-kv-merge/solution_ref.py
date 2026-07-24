import numpy as np


def merge_split_kv(partials):
    ms = np.stack([np.asarray(p[0], dtype=np.float64) for p in partials])   # (S, n)
    ls = np.stack([np.asarray(p[1], dtype=np.float64) for p in partials])   # (S, n)
    accs = np.stack([np.asarray(p[2], dtype=np.float64) for p in partials])  # (S, n, d)

    m_global = np.max(ms, axis=0)                    # (n,)
    correction = np.exp(ms - m_global[None, :])       # (S, n)

    l_global = np.sum(ls * correction, axis=0)        # (n,)
    acc_global = np.sum(accs * correction[:, :, None], axis=0)  # (n, d)

    return acc_global / l_global[:, None]
