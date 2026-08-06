import numpy as np
import math

def topk_deflation(A: np.ndarray, k: int):
    """
    Return the k largest eigenvalues and corresponding orthonormal eigenvectors
    of a real symmetric matrix A.
    """
    n = A.shape[0]
    A_val = [[float(A[i, j]) for j in range(n)] for i in range(n)]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(10000):
        max_val = -1.0
        p = 0
        q = 1
        for i in range(n):
            for j in range(i + 1, n):
                val = A_val[i][j]
                if val < 0:
                    val = -val
                if val > max_val:
                    max_val = val
                    p = i
                    q = j

        if max_val < 1e-15:
            break

        App = A_val[p][p]
        Aqq = A_val[q][q]
        Apq = A_val[p][q]

        theta = (Aqq - App) / (2.0 * Apq)
        if theta >= 0:
            t = 1.0 / (theta + math.sqrt(theta * theta + 1.0))
        else:
            t = -1.0 / (-theta + math.sqrt(theta * theta + 1.0))

        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c

        A_val[p][p] = c * c * App - 2.0 * s * c * Apq + s * s * Aqq
        A_val[q][q] = s * s * App + 2.0 * s * c * Apq + c * c * Aqq
        A_val[p][q] = 0.0
        A_val[q][p] = 0.0

        for i in range(n):
            if i != p and i != q:
                Aip = A_val[i][p]
                Aiq = A_val[i][q]
                A_val[i][p] = c * Aip - s * Aiq
                A_val[p][i] = A_val[i][p]
                A_val[i][q] = s * Aip + c * Aiq
                A_val[q][i] = A_val[i][q]

        for i in range(n):
            Vip = V[i][p]
            Viq = V[i][q]
            V[i][p] = c * Vip - s * Viq
            V[i][q] = s * Vip + c * Viq

    eigvals = []
    for i in range(n):
        eigvals.append((A_val[i][i], i))

    for i in range(n):
        for j in range(i + 1, n):
            if eigvals[j][0] > eigvals[i][0]:
                tmp = eigvals[i]
                eigvals[i] = eigvals[j]
                eigvals[j] = tmp

    out_vals = []
    for i in range(k):
        out_vals.append(eigvals[i][0])

    out_vecs = []
    for i in range(n):
        row = []
        for j in range(k):
            row.append(V[i][eigvals[j][1]])
        out_vecs.append(row)

    return np.array(out_vals, dtype=np.float64), np.array(out_vecs, dtype=np.float64)
