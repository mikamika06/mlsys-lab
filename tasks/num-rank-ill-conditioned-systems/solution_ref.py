import math
import numpy as np


def rank_by_condition(matrices):
    conditions = []
    for matrix in matrices:
        arr = np.asarray(matrix, dtype=np.float64)
        m, n = arr.shape
        if m == 0 or n == 0:
            conditions.append(0.0)
            continue

        A = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append(float(arr[i, j]))
            A.append(row)

        has_inf_or_nan = False
        for i in range(m):
            for j in range(n):
                val = A[i][j]
                if math.isnan(val) or math.isinf(val):
                    has_inf_or_nan = True
                    break
            if has_inf_or_nan:
                break

        if has_inf_or_nan:
            conditions.append(float("inf"))
            continue

        singular_values = []

        if m == 2 and n == 2:
            a, b = A[0][0], A[0][1]
            c, d = A[1][0], A[1][1]

            E = (a + d) / 2.0
            F = (a - d) / 2.0
            G = (c + b) / 2.0
            H = (c - b) / 2.0

            q = math.sqrt(E * E + H * H)
            r = math.sqrt(F * F + G * G)

            sv1 = q + r
            sv2 = math.abs(q - r) if hasattr(math, "abs") else abs(q - r)

            if sv1 >= sv2:
                singular_values = [sv1, sv2]
            else:
                singular_values = [sv2, sv1]
        else:
            AtA = []
            for i in range(n):
                row = []
                for j in range(n):
                    s = 0.0
                    for k in range(m):
                        s += A[k][i] * A[k][j]
                    row.append(s)
                AtA.append(row)

            V = []
            for i in range(n):
                row = []
                for j in range(n):
                    row.append(1.0 if i == j else 0.0)
                V.append(row)

            for _ in range(100):
                max_val = 0.0
                p, q = 0, 0
                for i in range(n):
                    for j in range(i + 1, n):
                        val = abs(AtA[i][j])
                        if val > max_val:
                            max_val = val
                            p, q = i, j

                if max_val < 1e-15:
                    break

                app = AtA[p][p]
                aqq = AtA[q][q]
                apq = AtA[p][q]

                if abs(apq) < 1e-15:
                    continue

                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))

                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                for i in range(n):
                    if i != p and i != q:
                        a_ip = AtA[i][p]
                        a_iq = AtA[i][q]
                        AtA[i][p] = c * a_ip - s * a_iq
                        AtA[p][i] = AtA[i][p]
                        AtA[i][q] = s * a_ip + c * a_iq
                        AtA[q][i] = AtA[i][q]

                AtA[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
                AtA[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
                AtA[p][q] = 0.0
                AtA[q][p] = 0.0

            eigenvalues = [max(0.0, AtA[i][i]) for i in range(n)]

            for i in range(n):
                for j in range(0, n - i - 1):
                    if eigenvalues[j] < eigenvalues[j + 1]:
                        eigenvalues[j], eigenvalues[j + 1] = (
                            eigenvalues[j + 1],
                            eigenvalues[j],
                        )

            singular_values = [math.sqrt(ev) for ev in eigenvalues]

        max_sv = singular_values[0]
        min_sv = singular_values[-1]

        if min_sv == 0.0:
            cond = float("inf")
        else:
            cond = max_sv / min_sv

        conditions.append(cond)

    indices = list(range(len(matrices)))
    for i in range(len(indices)):
        for j in range(0, len(indices) - i - 1):
            idx1 = indices[j]
            idx2 = indices[j + 1]
            if conditions[idx1] > conditions[idx2]:
                indices[j], indices[j + 1] = indices[j + 1], indices[j]

    return indices
