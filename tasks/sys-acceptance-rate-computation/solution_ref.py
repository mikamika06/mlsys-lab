import numpy as np

def acceptance_rate(target, draft):
    target = np.asarray(target, dtype=np.float64)
    draft = np.asarray(draft, dtype=np.float64)
    return np.sum(np.minimum(target, draft), axis=1)
