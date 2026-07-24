import numpy as np

def _ref(A, tol, max_iter):
    """Reference one-sided Jacobi SVD — compute singular values only."""
    A = A.astype(np.float64).copy()
    n = A.shape[0]
    G = A.T @ A
    for _ in range(max_iter):
        off_sq = 0.0
        best_val = 0.0
        best_i = best_j = 0
        for i in range(n):
            for j in range(i + 1, n):
                off_sq += 2.0 * G[i, j] ** 2
                if abs(G[i, j]) > best_val:
                    best_val = abs(G[i, j])
                    best_i, best_j = i, j
        if best_val < tol:
            break
        i, j = best_i, best_j
        tau = (G[j, j] - G[i, i]) / (2.0 * G[i, j])
        if abs(tau) < 1e-15:
            t = 1.0
        else:
            t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau ** 2))
        c = 1.0 / np.sqrt(1.0 + t ** 2)
        s = t * c
        col_i = A[:, i].copy()
        col_j = A[:, j].copy()
        A[:, i] = c * col_i - s * col_j
        A[:, j] = s * col_i + c * col_j
        G_i = G[i, :].copy()
        G_j = G[j, :].copy()
        G[i, :] = c * G_i - s * G_j
        G[:, i] = G[i, :]
        G[j, :] = s * G_i + c * G_j
        G[:, j] = G[j, :]
        G[i, i] = c ** 2 * G_i[i] - 2.0 * s * c * G_i[j] + s ** 2 * G_j[j]
        G[j, j] = s ** 2 * G_i[i] + 2.0 * s * c * G_i[j] + c ** 2 * G_j[j]
        G[i, j] = 0.0
        G[j, i] = 0.0
    return np.sort(np.sqrt(np.maximum(np.diag(G), 0.0)))[::-1]

def grade(sol, fx) -> dict:
    matrices = [
        np.array([[2.0, 1.0, 0.0],
                  [1.0, 3.0, 1.0],
                  [0.0, 1.0, 2.0]]),
        np.array([[4.0, 1.0, 0.0, 0.5],
                  [1.0, 3.0, 1.0, 0.0],
                  [0.0, 1.0, 2.0, 1.0],
                  [0.5, 0.0, 1.0, 5.0]]),
        np.array([[7.6, -3.6,  4.0,  6.0,  8.0],
                  [-3.6, 7.4,  6.0,  4.0,  2.0],
                  [4.0,  6.0,  6.5,  3.0, -1.0],
                  [6.0,  4.0,  3.0,  3.4, -2.0],
                  [8.0,  2.0, -1.0, -2.0,  2.5]]),
        np.array([[11.25,  3.75, -3.75,  3.75,  0.0],
                  [3.75,  11.25,  3.75, -3.75,  0.0],
                  [-3.75,  3.75, 11.25,  3.75,  0.0],
                  [3.75, -3.75,  3.75, 11.25,  0.0],
                  [0.0,   0.0,   0.0,   0.0,  1.0]]),
    ]
    rng = np.random.RandomState(42)
    Ar = rng.randn(6, 6)
    matrices.append(Ar @ Ar.T + 0.1 * np.eye(6))

    worst_rel = 0.0
    for A in matrices:
        try:
            got = np.array(sol.one_sided_jacobi_svd(A), dtype=np.float64)
            ref = _ref(A, tol=1e-12, max_iter=1000)
            got_sorted = np.sort(got)[::-1]
            rel = float(np.linalg.norm(got_sorted - ref) / (np.linalg.norm(ref) + 1e-12))
        except Exception:
            return {"rel_err": 1.0}
        worst_rel = max(worst_rel, rel)
    return {"rel_err": worst_rel}
