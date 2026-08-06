import math

def zca_whitening(X: list[list[float]]) -> list[list[float]]:
    n_samples = len(X)
    n_features = len(X[0])

    mean = [0.0] * n_features
    for i in range(n_samples):
        for j in range(n_features):
            mean[j] += float(X[i][j])
    for j in range(n_features):
        mean[j] /= n_samples

    X_centered = [[0.0] * n_features for _ in range(n_samples)]
    for i in range(n_samples):
        for j in range(n_features):
            X_centered[i][j] = float(X[i][j]) - mean[j]

    cov = [[0.0] * n_features for _ in range(n_features)]
    for j in range(n_features):
        for k in range(n_features):
            acc = 0.0
            for i in range(n_samples):
                acc += X_centered[i][j] * X_centered[i][k]
            cov[j][k] = acc / (n_samples - 1)

    A = [[cov[j][k] for k in range(n_features)] for j in range(n_features)]
    V = [[1.0 if j == k else 0.0 for k in range(n_features)] for j in range(n_features)]

    for _ in range(100):
        off_diag = 0.0
        for p in range(n_features):
            for q in range(p + 1, n_features):
                off_diag += abs(A[p][q])
        if off_diag < 1e-15:
            break

        for p in range(n_features):
            for q in range(p + 1, n_features):
                if abs(A[p][q]) < 1e-15:
                    continue
                tau = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                A_pp = A[p][p]
                A_qq = A[q][q]
                A_pq = A[p][q]

                A[p][p] = A_pp - t * A_pq
                A[q][q] = A_qq + t * A_pq
                A[p][q] = 0.0
                A[q][p] = 0.0

                for k in range(n_features):
                    if k != p and k != q:
                        a_kp = A[k][p]
                        a_kq = A[k][q]
                        A[k][p] = c * a_kp - s * a_kq
                        A[p][k] = A[k][p]
                        A[k][q] = s * a_kp + c * a_kq
                        A[q][k] = A[k][q]

                for k in range(n_features):
                    v_kp = V[k][p]
                    v_kq = V[k][q]
                    V[k][p] = c * v_kp - s * v_kq
                    V[k][q] = s * v_kp + c * v_kq

    inv_sqrt_eigvals = [1.0 / math.sqrt(A[k][k]) for k in range(n_features)]

    W_zca = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(n_features):
            acc = 0.0
            for k in range(n_features):
                acc += V[i][k] * inv_sqrt_eigvals[k] * V[j][k]
            W_zca[i][j] = acc

    X_whitened = [[0.0] * n_features for _ in range(n_samples)]
    for i in range(n_samples):
        for j in range(n_features):
            acc = 0.0
            for k in range(n_features):
                acc += X_centered[i][k] * W_zca[k][j]
            X_whitened[i][j] = acc

    return X_whitened
