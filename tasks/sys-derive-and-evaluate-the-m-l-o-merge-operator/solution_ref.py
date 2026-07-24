import numpy as np


def merge_mlo(state1, state2):
    m1, l1, o1 = state1
    m2, l2, o2 = state2

    m = max(float(m1), float(m2))
    a1 = np.exp(float(m1) - m)
    a2 = np.exp(float(m2) - m)

    l = float(l1) * a1 + float(l2) * a2
    o = np.asarray(o1, dtype=np.float64) * a1 + np.asarray(o2, dtype=np.float64) * a2

    return m, l, o
