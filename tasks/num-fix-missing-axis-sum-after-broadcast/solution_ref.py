import numpy as np

def broadcast_add(a, b):
    """Forward: c = a + b (a: (n,m), b: (m,)). Returns (c, backward).

    Backward takes dc (same shape as c) and returns (da, db).
    """
    n, m = a.shape
    c = np.empty((n, m), dtype=a.dtype)
    for i in range(n):
        for j in range(m):
            c[i, j] = a[i, j] + b[j]

    def backward(dc):
        da = np.empty((n, m), dtype=dc.dtype)
        for i in range(n):
            for j in range(m):
                da[i, j] = dc[i, j]

        db = np.zeros(m, dtype=dc.dtype)
        for i in range(n):
            for j in range(m):
                db[j] += dc[i, j]

        return da, db

    return c, backward
