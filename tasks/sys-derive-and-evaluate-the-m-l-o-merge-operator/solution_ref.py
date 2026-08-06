import math
import numpy as np


def merge_mlo(state1, state2):
    m1, l1, o1 = state1
    m2, l2, o2 = state2

    mf1 = float(m1)
    mf2 = float(m2)
    m = mf1 if mf1 > mf2 else mf2
    a1 = math.exp(mf1 - m)
    a2 = math.exp(mf2 - m)

    l = float(l1) * a1 + float(l2) * a2
    
    o1_arr = np.asarray(o1, dtype=np.float64)
    o2_arr = np.asarray(o2, dtype=np.float64)
    
    o = np.empty_like(o1_arr)
    for i in range(o1_arr.shape[0]):
        o[i] = o1_arr[i] * a1 + o2_arr[i] * a2

    return m, l, o
