import math
import numpy as np


def reconstruct_from_svd(A: np.ndarray) -> np.ndarray:
    """Reconstruct matrix from SVD."""
    A_arr = np.asarray(A, dtype=np.float64)
    m = A_arr.shape[0]
    n = A_arr.shape[1]

    A_list = [[A_arr[i, j] for j in range(n)] for i in range(m)]

    if n <= m:
        d = n
        B = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(d):
                acc = 0.0
                for k in range(m):
                    acc += A_list[k][i] * A_list[k][j]
                B[i][j] = acc

        V = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]

        for _ in range(100):
            max_val = 0.0
            p, q = 0, 1
            for i in range(d):
                for j in range(i + 1, d):
                    val = abs(B[i][j])
                    if val > max_val:
                        max_val = val
                        p, q = i, j
            if max_val < 1e-15:
                break

            app = B[p][p]
            aqq = B[q][q]
            apq = B[p][q]

            if abs(apq) < 1e-15:
                continue

            phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
            c = math.cos(phi)
            s_tr = math.sin(phi)

            for r in range(d):
                if r != p and r != q:
                    brp = B[r][p]
                    brq = B[r][q]
                    B[r][p] = c * brp - s_tr * brq
                    B[p][r] = B[r][p]
                    B[r][q] = s_tr * brp + c * brq
                    B[q][r] = B[r][q]

            B[p][p] = c * c * app - 2.0 * s_tr * c * apq + s_tr * s_tr * aqq
            B[q][q] = s_tr * s_tr * app + 2.0 * s_tr * c * apq + c * c * aqq
            B[p][q] = 0.0
            B[q][p] = 0.0

            for r in range(d):
                vrp = V[r][p]
                vrq = V[r][q]
                V[r][p] = c * vrp - s_tr * vrq
                V[r][q] = s_tr * vrp + c * vrq

        vecs = [[V[r][i] for r in range(d)] for i in range(d)]

        W = []
        for i in range(d):
            w = [0.0] * m
            for r in range(m):
                acc = 0.0
                for j in range(d):
                    acc += A_list[r][j] * vecs[i][j]
                w[r] = acc
            W.append(w)

        recon_list = [[0.0] * n for _ in range(m)]
        for r in range(m):
            for c_idx in range(n):
                acc = 0.0
                for i in range(d):
                    acc += W[i][r] * vecs[i][c_idx]
                recon_list[r][c_idx] = acc

    else:
        d = m
        B = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(d):
                acc = 0.0
                for k in range(n):
                    acc += A_list[i][k] * A_list[j][k]
                B[i][j] = acc

        U_mat = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]

        for _ in range(100):
            max_val = 0.0
            p, q = 0, 1
            for i in range(d):
                for j in range(i + 1, d):
                    val = abs(B[i][j])
                    if val > max_val:
                        max_val = val
                        p, q = i, j
            if max_val < 1e-15:
                break

            app = B[p][p]
            aqq = B[q][q]
            apq = B[p][q]

            if abs(apq) < 1e-15:
                continue

            phi = 0.5 * math.atan2(2.0 * apq, aqq - app)
            c = math.cos(phi)
            s_tr = math.sin(phi)

            for r in range(d):
                if r != p and r != q:
                    brp = B[r][p]
                    brq = B[r][q]
                    B[r][p] = c * brp - s_tr * brq
                    B[p][r] = B[r][p]
                    B[r][q] = s_tr * brp + c * brq
                    B[q][r] = B[r][q]

            B[p][p] = c * c * app - 2.0 * s_tr * c * apq + s_tr * s_tr * aqq
            B[q][q] = s_tr * s_tr * app + 2.0 * s_tr * c * apq + c * c * aqq
            B[p][q] = 0.0
            B[q][p] = 0.0

            for r in range(d):
                vrp = U_mat[r][p]
                vrq = U_mat[r][q]
                U_mat[r][p] = c * vrp - s_tr * vrq
                U_mat[r][q] = s_tr * vrp + c * vrq

        vecs = [[U_mat[r][i] for r in range(d)] for i in range(d)]

        Y = []
        for i in range(d):
            y = [0.0] * n
            for c_idx in range(n):
                acc = 0.0
                for r in range(m):
                    acc += vecs[i][r] * A_list[r][c_idx]
                y[c_idx] = acc
            Y.append(y)

        recon_list = [[0.0] * n for _ in range(m)]
        for r in range(m):
            for c_idx in range(n):
                acc = 0.0
                for i in range(d):
                    acc += vecs[i][r] * Y[i][c_idx]
                recon_list[r][c_idx] = acc

    return np.array(recon_list, dtype=np.float64)
