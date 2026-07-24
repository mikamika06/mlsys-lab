import numpy as np

def broadcast_add(a, b):
    """Forward: c = a + b (a: (n,m), b: (m,)). Returns (c, backward).

    Backward takes dc and returns (da, db).  BUGGY — fix the backward pass.
    """
    c = a + b

    def backward(dc):
        da = dc
        db = dc  # BUG: missing np.sum(dc, axis=0) over the broadcast axis
        return da, db

    return c, backward
