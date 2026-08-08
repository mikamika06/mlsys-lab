import numpy as np

def conditional_branch_eager(pred, x, y):
    if bool(pred):
        return x * 2.0
    else:
        return y + 1.0

def conditional_branch_cond(pred, x, y):
    if bool(pred):
        res = x * 2.0
    else:
        res = y + 1.0
    return np.asarray(res, dtype=np.float32)
