import math
import numpy as np


def mds_from_distances(D2: np.ndarray, k: int) -> np.ndarray:
    """
    Classical Multidimensional Scaling.

    Parameters
    ----------
    D2 : np.ndarray
        Squared Euclidean distance matrix of shape (n, n).
    k : int
        Target dimensionality of the embedding.

    Returns
    -------
    X : np.ndarray
        Coordinates of shape (n, k) in float64.
    """
    if D2.ndim != 2 or D2.shape[0] != D2.shape[1]:
        raise ValueError("D2 must be a square matrix")
    n = D2.shape[0]

    row_sums = [0.0] * n
    col_sums = [0.0] * n
    total_sum = 0.0

    for i in range(n):
        for j in range(n):
            val = float(D2[i, j])
            row_sums[i] += val
            col_sums[j] += val
            total_sum += val

    grand_mean = total_sum / (n * n)

    B = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_m = row_sums[i] / n
        for j in range(n):
            col_m = col_sums[j] / n
            B[i][j] = -0.5 * (float(D2[i, j]) - row_m - col_m + grand_mean)

    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(100):
        off_diag = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                val = B[p][q]
                off_diag += val if val >= 0 else -val
        if off_diag < 1e-15:
            break

        for p in range(n - 1):
            for q in range(p + 1, n):
                app = B[p][p]
                aqq = B[q][q]
                apq = B[p][q]
                abs_apq = apq if apq >= 0 else -apq
                if abs_apq < 1e-15:
                    continue

                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))

                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                B[p][p] = app - t * apq
                B[q][q] = aqq + t * apq
                B[p][q] = 0.0
                B[q][p] = 0.0

                for r in range(n):
                    if r != p and r != q:
                        arp = B[r][p]
                        arq = B[r][q]
                        brp = c * arp - s * arq
                        brq = s * arp + c * arq
                        B[r][p] = brp
                        B[p][r] = brp
                        B[r][q] = brq
                        B[q][r] = brq

                for r in range(n):
                    vrp = V[r][p]
                    vrq = V[r][q]
                    V[r][p] = c * vrp - s * vrq
                    V[r][q] = s * vrp + c * vrq

    eigvals = [B[i][i] for i in range(n)]

    order = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if eigvals[order[i]] > eigvals[order[j]]:
                order[i], order[j] = order[j], order[i]
    idx = order[::-1]

    k_eff = k if k < n else n
    idx_k = idx[:k_eff]

    X = np.empty((n, k_eff), dtype=np.float64)
    for i in range(n):
        for m in range(k_eff):
            eigenval = eigvals[idx_k[m]]
            val = eigenval if eigenval > 0.0 else 0.0
            lam_sqrt = math.sqrt(val)
            X[i, m] = V[i][idx_k[m]] * lam_sqrt

    return X
