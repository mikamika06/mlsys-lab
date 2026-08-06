import math
import numpy as np


def pca_slice_error(X: np.ndarray, k: int) -> float:
    """Squared Frobenius error of projecting X onto the top-k eigenvectors
    of the Gram matrix G = X^T X (no centering): ||X - X Q_k Q_k^T||_F^2.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    G = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            acc = 0.0
            for r in range(n):
                acc += X[r, i] * X[r, j]
            G[i, j] = acc

    V = np.eye(d, dtype=np.float64)
    A = G.copy()
    for _ in range(100):
        max_val = 0.0
        for p in range(d):
            for q in range(p + 1, d):
                if math.fabs(A[p, q]) > max_val:
                    max_val = math.fabs(A[p, q])
        if max_val < 1e-15:
            break
        for p in range(d):
            for q in range(p + 1, d):
                if math.fabs(A[p, q]) < 1e-12:
                    continue
                app = A[p, p]
                aqq = A[q, q]
                apq = A[p, q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                A[p, p] = app - t * apq
                A[q, q] = aqq + t * apq
                A[p, q] = 0.0
                A[q, p] = 0.0
                for r in range(d):
                    if r != p and r != q:
                        arp = A[r, p]
                        arq = A[r, q]
                        A[r, p] = c * arp - s * arq
                        A[p, r] = A[r, p]
                        A[r, q] = s * arp + c * arq
                        A[q, r] = A[r, q]
                for r in range(d):
                    vrp = V[r, p]
                    vrq = V[r, q]
                    V[r, p] = c * vrp - s * vrq
                    V[r, q] = s * vrp + c * vrq

    eigvals = [A[i, i] for i in range(d)]
    indices = sorted(range(d), key=lambda i: eigvals[i])
    eigvecs_asc = np.zeros((d, d), dtype=np.float64)
    for new_idx, old_idx in enumerate(indices):
        for r in range(d):
            eigvecs_asc[r, new_idx] = V[r, old_idx]

    if k <= 0:
        Xrec = np.zeros((n, d), dtype=np.float64)
    else:
        Temp = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            for j in range(k):
                acc = 0.0
                col_idx = (d - k) + j
                for r in range(d):
                    acc += X[i, r] * eigvecs_asc[r, col_idx]
                Temp[i, j] = acc
        Xrec = np.zeros((n, d), dtype=np.float64)
        for i in range(n):
            for j in range(d):
                acc = 0.0
                for r in range(k):
                    col_idx = (d - k) + r
                    acc += Temp[i, r] * eigvecs_asc[j, col_idx]
                Xrec[i, j] = acc

    total_sum = 0.0
    for i in range(n):
        for j in range(d):
            diff = X[i, j] - Xrec[i, j]
            total_sum += diff * diff

    return float(total_sum)
