import numpy as np


def fix_qzeros(qzeros, group_size):
    out = qzeros.copy()
    out = np.clip(out + 1, 0, None)
    return out
