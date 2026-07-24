import numpy as np


def merge_split_kv(partials):
    m = max(p[0] for p in partials)

    l = 0.0
    numerator = None
    for m_i, l_i, o_i in partials:
        scale = np.exp(m_i - m)
        l += scale * l_i
        term = scale * np.asarray(o_i, dtype=np.float64)
        if numerator is None:
            numerator = term
        else:
            numerator = numerator + term

    return numerator / l
