import math
import numpy as np


def optimal_momentum_beta(A: np.ndarray) -> float:
    """
    Compute the optimal momentum coefficient for gradient descent on a quadratic
    with Hessian A, given by ((sqrt(kappa)-1)/(sqrt(kappa)+1))^2,
    where kappa is the condition number of A.
    """
    if not isinstance(A, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            val_ij = float(A[i, j])
            val_ji = float(A[j, i])
            if abs(val_ij - val_ji) > (1e-8 + 1e-5 * abs(val_ji)):
                raise ValueError("A must be symmetric.")

    M = [[float(A[i, j]) for j in range(n)] for i in range(n)]

    for _ in range(100):
        off_diag = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off_diag += abs(M[i][j])
        if off_diag < 1e-15:
            break

        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = M[p][q]
                if abs(apq) < 1e-15:
                    continue
                app = M[p][p]
                aqq = M[q][q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                M[p][p] = app - t * apq
                M[q][q] = aqq + t * apq
                M[p][q] = 0.0
                M[q][p] = 0.0

                for r in range(n):
                    if r != p and r != q:
                        arp = M[r][p]
                        arq = M[r][q]
                        M[r][p] = c * arp - s * arq
                        M[p][r] = M[r][p]
                        M[r][q] = s * arp + c * arq
                        M[q][r] = M[r][q]

    eigs = [M[i][i] for i in range(n)]

    for i in range(len(eigs)):
        min_idx = i
        for j in range(i + 1, len(eigs)):
            if eigs[j] < eigs[min_idx]:
                min_idx = j
        eigs[i], eigs[min_idx] = eigs[min_idx], eigs[i]

    for val in eigs:
        if val <= 0.0:
            raise ValueError("A must be positive-definite.")

    kappa = eigs[-1] / eigs[0]
    beta = ((math.sqrt(kappa) - 1.0) / (math.sqrt(kappa) + 1.0)) ** 2
    return float(beta)
