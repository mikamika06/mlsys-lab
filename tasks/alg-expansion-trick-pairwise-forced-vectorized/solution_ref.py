import numpy as np

def _compute(A, n, d):
    g = [0.0] * n
    for i in range(n):
        s = 0.0
        for k in range(d):
            v = float(A[i, k])
            s += v * v
        g[i] = s
    
    out = []
    for i in range(n):
        row = []
        for j in range(n):
            s = 0.0
            for k in range(d):
                s += float(A[i, k]) * float(A[j, k])
            row.append(g[i] + g[j] - 2.0 * s)
        out.append(row)
    return out

def pairwise_sq_dists(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    d = A.shape[1]
    return np.array(_compute(A, n, d), dtype=np.float64)
