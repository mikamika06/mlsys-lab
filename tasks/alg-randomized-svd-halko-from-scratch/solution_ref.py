import math
import numpy as np

def randomized_svd(A: np.ndarray, k: int, seed: int):
    """
    Randomized SVD implemented with basic loops and math functions.
    """
    m, n = A.shape
    rng = np.random.default_rng(seed)

    p = 5 if n - k > 5 else n - k
    q = k + p

    omega = rng.normal(size=(n, q))
    Y = np.zeros((m, q))
    for i in range(m):
        for j in range(q):
            val = 0.0
            for x in range(n):
                val += A[i, x] * omega[x, j]
            Y[i, j] = val

    Q = np.zeros((m, q))
    for i in range(m):
        for j in range(q):
            Q[i, j] = Y[i, j]

    for j in range(q):
        norm_sq = 0.0
        for i in range(m):
            norm_sq += Q[i, j] ** 2
        norm = math.sqrt(norm_sq)
        if norm > 0:
            for i in range(m):
                Q[i, j] /= norm
        for c in range(j + 1, q):
            dot = 0.0
            for i in range(m):
                dot += Q[i, j] * Q[i, c]
            for i in range(m):
                Q[i, c] -= dot * Q[i, j]

    B = np.zeros((q, n))
    for i in range(q):
        for j in range(n):
            val = 0.0
            for x in range(m):
                val += Q[x, i] * A[x, j]
            B[i, j] = val

    M = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            val = 0.0
            for x in range(n):
                val += B[i, x] * B[j, x]
            M[i, j] = val

    V = np.zeros((q, q))
    for i in range(q):
        V[i, i] = 1.0

    for _ in range(1000):
        max_val = 0.0
        p_idx = 0
        q_idx = 0
        for i in range(q):
            for j in range(i + 1, q):
                val = M[i, j] if M[i, j] > 0 else -M[i, j]
                if val > max_val:
                    max_val = val
                    p_idx = i
                    q_idx = j

        if max_val < 1e-13:
            break

        mpp = M[p_idx, p_idx]
        mqq = M[q_idx, q_idx]
        mpq = M[p_idx, q_idx]

        phi = 0.5 * math.atan2(2.0 * mpq, mpp - mqq)
        c = math.cos(phi)
        s = math.sin(phi)

        for i in range(q):
            if i != p_idx and i != q_idx:
                mip = M[i, p_idx]
                miq = M[i, q_idx]
                M[i, p_idx] = c * mip + s * miq
                M[p_idx, i] = M[i, p_idx]
                M[i, q_idx] = -s * mip + c * miq
                M[q_idx, i] = M[i, q_idx]

        M[p_idx, p_idx] = c * c * mpp + 2.0 * s * c * mpq + s * s * mqq
        M[q_idx, q_idx] = s * s * mpp - 2.0 * s * c * mpq + c * c * mqq
        M[p_idx, q_idx] = 0.0
        M[q_idx, p_idx] = 0.0

        for i in range(q):
            vip = V[i, p_idx]
            viq = V[i, q_idx]
            V[i, p_idx] = c * vip + s * viq
            V[i, q_idx] = -s * vip + c * viq

    eigvals = [M[i, i] for i in range(q)]
    indices = list(range(q))
    for i in range(q):
        for j in range(i + 1, q):
            if eigvals[indices[j]] > eigvals[indices[i]]:
                indices[i], indices[j] = indices[j], indices[i]

    S = np.zeros(k)
    Ub = np.zeros((q, k))
    for i in range(k):
        idx = indices[i]
        val = eigvals[idx]
        S[i] = math.sqrt(val) if val > 0 else 0.0
        for j in range(q):
            Ub[j, i] = V[j, idx]

    Vt = np.zeros((k, n))
    for i in range(k):
        if S[i] > 1e-12:
            for j in range(n):
                dot = 0.0
                for x in range(q):
                    dot += Ub[x, i] * B[x, j]
                Vt[i, j] = dot / S[i]
        else:
            Vt[i, i] = 1.0

    final_U = np.zeros((m, k))
    for i in range(m):
        for j in range(k):
            val = 0.0
            for x in range(q):
                val += Q[i, x] * Ub[x, j]
            final_U[i, j] = val

    return final_U, S, Vt
