import numpy as np
import math

def pca_svd(X: np.ndarray, k: int) -> np.ndarray:
    """PCA using SVD via Jacobi eigendecomposition written by hand."""
    N = X.shape[0]
    D = X.shape[1]

    mean = [0.0] * D
    for j in range(D):
        s = 0.0
        for i in range(N):
            s += float(X[i, j])
        mean[j] = s / N

    centered = [[0.0] * D for _ in range(N)]
    for i in range(N):
        for j in range(D):
            centered[i][j] = float(X[i, j]) - mean[j]

    C = [[0.0] * D for _ in range(D)]
    for i in range(D):
        for j in range(D):
            s = 0.0
            for m in range(N):
                s += centered[m][i] * centered[m][j]
            C[i][j] = s

    V = [[0.0] * D for _ in range(D)]
    for i in range(D):
        V[i][i] = 1.0

    A = [[C[i][j] for j in range(D)] for i in range(D)]

    for _ in range(100):
        max_val = 0.0
        p, q = 0, 1
        for i in range(D):
            for j in range(i + 1, D):
                val = A[i][j] if A[i][j] >= 0 else -A[i][j]
                if val > max_val:
                    max_val = val
                    p, q = i, j

        if max_val < 1e-15:
            break

        app = A[p][p]
        aqq = A[q][q]
        apq = A[p][q]

        theta = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c = math.cos(theta)
        s = math.sin(theta)

        for r in range(D):
            if r != p and r != q:
                arp = A[r][p]
                arq = A[r][q]
                A[r][p] = c * arp - s * arq
                A[p][r] = A[r][p]
                A[r][q] = s * arp + c * arq
                A[q][r] = A[r][q]

        A[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
        A[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
        A[p][q] = 0.0
        A[q][p] = 0.0

        for r in range(D):
            vrp = V[r][p]
            vrq = V[r][q]
            V[r][p] = c * vrp - s * vrq
            V[r][q] = s * vrp + c * vrq

    evals = [A[i][i] for i in range(D)]
    indices = list(range(D))
    for i in range(D):
        for j in range(i + 1, D):
            if evals[indices[i]] < evals[indices[j]]:
                indices[i], indices[j] = indices[j], indices[i]

    out = np.zeros((N, k), dtype=np.float64)
    for i in range(N):
        for j in range(k):
            col_idx = indices[j]
            s = 0.0
            for m in range(D):
                s += centered[i][m] * V[m][col_idx]
            out[i, j] = s

    return out
