import math
import numpy as np


def svd_singular_values(A: np.ndarray) -> np.ndarray:
    """Singular values of A via the eigendecomposition of the Gram matrix A^T A."""
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    k = min(m, n)
    G = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for l in range(m):
                s += A[l, i] * A[l, j]
            G[i][j] = s
    S = [row[:] for row in G]
    for _ in range(100):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += S[p][q] * S[p][q]
        if off < 1e-24:
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = S[p][q]
                if abs(apq) < 1e-15:
                    continue
                app = S[p][p]
                aqq = S[q][q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s_val = t * c
                S[p][p] = c * c * app - 2.0 * s_val * c * apq + s_val * s_val * aqq
                S[q][q] = s_val * s_val * app + 2.0 * s_val * c * apq + c * c * aqq
                S[p][q] = 0.0
                S[q][p] = 0.0
                for r in range(n):
                    if r != p and r != q:
                        arp = S[r][p]
                        arq = S[r][q]
                        S[r][p] = c * arp - s_val * arq
                        S[p][r] = S[r][p]
                        S[r][q] = s_val * arp + c * arq
                        S[q][r] = S[r][q]
    w = [max(0.0, S[i][i]) for i in range(n)]
    vals = sorted([math.sqrt(x) for x in w], reverse=True)
    return np.asarray(vals[:k], dtype=np.float64)
