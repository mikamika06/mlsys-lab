import math
import numpy as np


def low_rank_factors(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return thin factors (A, B) with A = U_k @ Sigma_k, B = V_k^T."""
    m = W.shape[0]
    n = W.shape[1]
    A = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            A[i, j] = float(W[i, j])

    V = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        V[i, i] = 1.0

    for _ in range(30):
        any_changed = False
        for i in range(n):
            for j in range(i + 1, n):
                alpha = 0.0
                beta = 0.0
                gamma = 0.0
                for r in range(m):
                    ari = A[r, i]
                    arj = A[r, j]
                    alpha += ari * ari
                    beta += arj * arj
                    gamma += ari * arj

                if math.fabs(gamma) < 1e-15 * math.sqrt(alpha * beta + 1e-300):
                    continue

                any_changed = True
                tau = (beta - alpha) / (2.0 * gamma)
                if math.fabs(tau) > 1e15:
                    t = 0.5 / tau
                elif tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))

                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                for r in range(m):
                    ari = A[r, i]
                    arj = A[r, j]
                    A[r, i] = c * ari - s * arj
                    A[r, j] = s * ari + c * arj

                for r in range(n):
                    vri = V[r, i]
                    vrj = V[r, j]
                    V[r, i] = c * vri - s * vrj
                    V[r, j] = s * vri + c * vrj

        if not any_changed:
            break

    norms = [0.0] * n
    for j in range(n):
        sum_sq = 0.0
        for r in range(m):
            sum_sq += A[r, j] * A[r, j]
        norms[j] = math.sqrt(sum_sq)

    order = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if norms[order[j]] > norms[order[i]]:
                order[i], order[j] = order[j], order[i]

    A_out = np.zeros((m, k), dtype=np.float64)
    B_out = np.zeros((k, n), dtype=np.float64)

    limit = k if k < n else n
    for c in range(limit):
        idx = order[c]
        for r in range(m):
            A_out[r, c] = A[r, idx]
        for col in range(n):
            B_out[c, col] = V[col, idx]

    return A_out, B_out


def low_rank_reconstruct(W: np.ndarray, k: int) -> np.ndarray:
    """Return the optimal rank-k approximation A @ B of W."""
    A, B = low_rank_factors(W, k)
    m = A.shape[0]
    k_dim = A.shape[1]
    n = B.shape[1]
    out = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k_dim):
                s += A[i, p] * B[p, j]
            out[i, j] = s
    return out
