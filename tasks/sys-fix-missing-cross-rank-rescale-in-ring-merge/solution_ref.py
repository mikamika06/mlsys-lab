import math
import numpy as np


def ring_merge(partials):
    ms = [np.asarray(p[0], dtype=np.float64) for p in partials]
    ls = [np.asarray(p[1], dtype=np.float64) for p in partials]
    accs = [np.asarray(p[2], dtype=np.float64) for p in partials]

    num_rows = ms[0].shape[0]
    dim = accs[0].shape[1]

    m_new = np.empty_like(ms[0])
    for i in range(num_rows):
        max_val = ms[0][i]
        for m in ms[1:]:
            if m[i] > max_val:
                max_val = m[i]
        m_new[i] = max_val

    total_l = np.zeros_like(m_new)
    total_a = np.zeros_like(accs[0], dtype=np.float64)

    for m, l, a in zip(ms, ls, accs):
        for i in range(num_rows):
            scale_i = math.exp(m[i] - m_new[i])
            total_l[i] += l[i] * scale_i
            for j in range(dim):
                total_a[i, j] += a[i, j] * scale_i

    res = np.empty_like(total_a)
    for i in range(num_rows):
        inv_l = 1.0 / total_l[i]
        for j in range(dim):
            res[i, j] = total_a[i, j] * inv_l

    return res
