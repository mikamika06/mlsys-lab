import math
import numpy as np


def hilbert_condition_numbers(ns):
    result = []
    for n in ns:
        A = [[1.0 / (float(i) + float(j) + 1.0) for j in range(n)] for i in range(n)]
        for sweep in range(30):
            changed = False
            for p in range(n):
                for q in range(p + 1, n):
                    alpha = 0.0
                    beta = 0.0
                    gamma = 0.0
                    for k in range(n):
                        akp = A[k][p]
                        akq = A[k][q]
                        alpha += akp * akp
                        beta += akq * akq
                        gamma += akp * akq
                    if gamma != 0.0 and gamma * gamma > 1e-32 * alpha * beta:
                        tau = (beta - alpha) / (2.0 * gamma)
                        if tau >= 0.0:
                            t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                        else:
                            t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                        c = 1.0 / math.sqrt(1.0 + t * t)
                        s = t * c
                        for k in range(n):
                            akp = A[k][p]
                            akq = A[k][q]
                            A[k][p] = c * akp - s * akq
                            A[k][q] = s * akp + c * akq
                        changed = True
            if not changed:
                break
        s_vals = []
        for j in range(n):
            col_sum = 0.0
            for k in range(n):
                col_sum += A[k][j] * A[k][j]
            s_vals.append(math.sqrt(col_sum))
        s_max = s_vals[0]
        s_min = s_vals[0]
        for val in s_vals[1:]:
            if val > s_max:
                s_max = val
            if val < s_min:
                s_min = val
        result.append(math.log10(s_max / s_min))
    return np.asarray(result, dtype=np.float64)
