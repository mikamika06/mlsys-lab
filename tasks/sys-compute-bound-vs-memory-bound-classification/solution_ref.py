import numpy as np

def classify_bound(ai: np.ndarray, balance: float) -> np.ndarray:
    ai = np.asarray(ai)
    res = []
    for x in ai.flat:
        if x > balance:
            res.append('compute-bound')
        else:
            res.append('memory-bound')
    return np.array(res).reshape(ai.shape)
