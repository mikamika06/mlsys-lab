import numpy as np

def adapt_to_flash(q, k, v):
    if q.ndim == 4:
        q_fa = np.transpose(q, (0, 2, 1, 3))
        k_fa = np.transpose(k, (0, 2, 1, 3))
        v_fa = np.transpose(v, (0, 2, 1, 3))
    else:
        q_fa = q
        k_fa = k
        v_fa = v
    return np.ascontiguousarray(q_fa), np.ascontiguousarray(k_fa), np.ascontiguousarray(v_fa)

def adapt_from_flash(out):
    if out.ndim == 4:
        return np.ascontiguousarray(np.transpose(out, (0, 2, 1, 3)))
    return out
