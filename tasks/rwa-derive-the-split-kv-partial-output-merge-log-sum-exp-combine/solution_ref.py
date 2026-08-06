import math
import numpy as np


def merge_split_kv(partials):
    m = max(p[0] for p in partials)

    l = 0.0
    numerator = None
    for m_i, l_i, o_i in partials:
        scale = math.exp(m_i - m)
        l += scale * l_i
        term = [scale * x for x in o_i]
        if numerator is None:
            numerator = term
        else:
            numerator = [n + t for n, t in zip(numerator, term)]

    return np.asarray(numerator, dtype=np.float64) / l
