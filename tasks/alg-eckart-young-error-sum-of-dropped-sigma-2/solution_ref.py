import math
import numpy as np


def eckart_young_errors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Eckart-Young approximation errors."""
    X = np.asarray(X, dtype=np.float64)
    m, n = X.shape
    r = min(m, n)

    if m >= n:
        B = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(m):
                    acc += X[k, i] * X[k, j]
                B[i, j] = acc

        n_dim = n
        V = np.zeros((n_dim, n_dim), dtype=np.float64)
        for i in range(n_dim):
            V[i, i] = 1.0

        D = np.zeros((n_dim, n_dim), dtype=np.float64)
        for i in range(n_dim):
            for j in range(n_dim):
                D[i, j] = B[i, j]

        for _ in range(100):
            max_off = 0.0
            for i in range(n_dim):
                for j in range(i + 1, n_dim):
                    if math.fabs(D[i, j]) > max_off:
                        max_off = math.fabs(D[i, j])
            if max_off < 1e-15:
                break

            for p in range(n_dim):
                for q in range(p + 1, n_dim):
                    apq = D[p, q]
                    if math.fabs(apq) > 1e-15:
                        app = D[p, p]
                        aqq = D[q, q]
                        theta = 0.5 * math.atan2(2.0 * apq, aqq - app)
                        c = math.cos(theta)
                        s = math.sin(theta)

                        old_app = app
                        old_aqq = aqq
                        old_apq = apq

                        for rd in range(n_dim):
                            if rd != p and rd != q:
                                drp = D[rd, p]
                                drq = D[rd, q]
                                val_p = c * drp - s * drq
                                val_q = s * drp + c * drq
                                D[rd, p] = val_p
                                D[p, rd] = val_p
                                D[rd, q] = val_q
                                D[q, rd] = val_q

                        D[p, p] = c * c * old_app - 2.0 * s * c * old_apq + s * s * old_aqq
                        D[q, q] = s * s * old_app + 2.0 * s * c * old_apq + c * c * old_aqq
                        D[p, q] = 0.0
                        D[q, p] = 0.0

                        for rd in range(n_dim):
                            vrp = V[rd, p]
                            vrq = V[rd, q]
                            V[rd, p] = c * vrp - s * vrq
                            V[rd, q] = s * vrp + c * vrq

        evals = []
        for i in range(n_dim):
            evals.append(D[i, i])

        indices = list(range(n_dim))
        for i in range(n_dim):
            for j in range(i + 1, n_dim):
                if evals[indices[j]] > evals[indices[i]]:
                    indices[i], indices[j] = indices[j], indices[i]

        s = np.zeros(n, dtype=np.float64)
        sorted_V = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            idx = indices[i]
            s[i] = math.sqrt(max(0.0, evals[idx]))
            for rd in range(n):
                sorted_V[rd, i] = V[rd, idx]

        u = np.zeros((m, n), dtype=np.float64)
        for i in range(n):
            si = s[i]
            if si > 1e-12:
                for rd in range(m):
                    acc = 0.0
                    for c in range(n):
                        acc += X[rd, c] * sorted_V[c, i]
                    u[rd, i] = acc / si
            else:
                u[i, i] = 1.0

        vt = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                vt[i, j] = sorted_V[j, i]

    else:
        B = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(m):
                acc = 0.0
                for k in range(n):
                    acc += X[i, k] * X[j, k]
                B[i, j] = acc

        n_dim = m
        V = np.zeros((n_dim, n_dim), dtype=np.float64)
        for i in range(n_dim):
            V[i, i] = 1.0

        D = np.zeros((n_dim, n_dim), dtype=np.float64)
        for i in range(n_dim):
            for j in range(n_dim):
                D[i, j] = B[i, j]

        for _ in range(100):
            max_off = 0.0
            for i in range(n_dim):
                for j in range(i + 1, n_dim):
                    if math.fabs(D[i, j]) > max_off:
                        max_off = math.fabs(D[i, j])
            if max_off < 1e-15:
                break

            for p in range(n_dim):
                for q in range(p + 1, n_dim):
                    apq = D[p, q]
                    if math.fabs(apq) > 1e-15:
                        app = D[p, p]
                        aqq = D[q, q]
                        theta = 0.5 * math.atan2(2.0 * apq, aqq - app)
                        c = math.cos(theta)
                        s_rot = math.sin(theta)

                        old_app = app
                        old_aqq = aqq
                        old_apq = apq

                        for rd in range(n_dim):
                            if rd != p and rd != q:
                                drp = D[rd, p]
                                drq = D[rd, q]
                                val_p = c * drp - s_rot * drq
                                val_q = s_rot * drp + c * drq
                                D[rd, p] = val_p
                                D[p, rd] = val_p
                                D[rd, q] = val_q
                                D[q, rd] = val_q

                        D[p, p] = c * c * old_app - 2.0 * s_rot * c * old_apq + s_rot * s_rot * old_aqq
                        D[q, q] = s_rot * s_rot * old_app + 2.0 * s_rot * c * old_apq + c * c * old_aqq
                        D[p, q] = 0.0
                        D[q, p] = 0.0

                        for rd in range(n_dim):
                            vrp = V[rd, p]
                            vrq = V[rd, q]
                            V[rd, p] = c * vrp - s_rot * vrq
                            V[rd, q] = s_rot * vrp + c * vrq

        evals = []
        for i in range(n_dim):
            evals.append(D[i, i])

        indices = list(range(n_dim))
        for i in range(n_dim):
            for j in range(i + 1, n_dim):
                if evals[indices[j]] > evals[indices[i]]:
                    indices[i], indices[j] = indices[j], indices[i]

        s = np.zeros(m, dtype=np.float64)
        sorted_U = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            idx = indices[i]
            s[i] = math.sqrt(max(0.0, evals[idx]))
            for rd in range(m):
                sorted_U[rd, i] = V[rd, idx]

        u = sorted_U

        vt = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            si = s[i]
            if si > 1e-12:
                for c in range(n):
                    acc = 0.0
                    for rd in range(m):
                        acc += u[rd, i] * X[rd, c]
                    vt[i, c] = acc / si
            else:
                vt[i, i] = 1.0

    direct = []
    theorem = []
    for k in range(r + 1):
        xk = np.zeros((m, n), dtype=np.float64)
        if k > 0:
            for i in range(m):
                for j in range(n):
                    acc = 0.0
                    for l in range(k):
                        acc += (u[i, l] * s[l]) * vt[l, j]
                    xk[i, j] = acc

        d_acc = 0.0
        for i in range(m):
            for j in range(n):
                diff = X[i, j] - xk[i, j]
                d_acc += diff * diff
        direct.append(d_acc)

        t_acc = 0.0
        for l in range(k, r):
            t_acc += s[l] * s[l]
        theorem.append(t_acc)

    return np.asarray(direct, dtype=np.float64), np.asarray(theorem, dtype=np.float64)
