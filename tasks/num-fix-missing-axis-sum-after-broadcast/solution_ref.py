import numpy as np

def broadcast_add(a, b):
    """Forward: c = a + b (a: (n,m), b: (m,)). Returns (c, backward).

    Backward takes dc (same shape as c) and returns (da, db).
    """
    c = a + b

    def backward(dc):
        da = dc
        db = np.sum(dc, axis=0)
        return da, db

    return c, backward
