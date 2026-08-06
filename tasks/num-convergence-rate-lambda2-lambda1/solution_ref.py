import math
import numpy as np


def estimate_convergence_rate(A):
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    x0 = np.zeros(n, dtype=np.float64)
    sq_sum = 0.0
    for i in range(n):
        val = float(i + 1)
        x0[i] = val
        sq_sum += val * val

    norm_x0 = math.sqrt(sq_sum)
    for i in range(n):
        x0[i] /= norm_x0

    v = np.zeros(n, dtype=np.float64)
    for i in range(n):
        v[i] = x0[i]

    for _ in range(50):
        v_next = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * v[j]
            v_next[i] = s

        sq_sum = 0.0
        for i in range(n):
            sq_sum += v_next[i] * v_next[i]
        norm_v = math.sqrt(sq_sum)

        for i in range(n):
            v[i] = v_next[i] / norm_v

    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x[i] = x0[i]

    errors = []

    for _ in range(50):
        x_next = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * x[j]
            x_next[i] = s

        sq_sum = 0.0
        for i in range(n):
            sq_sum += x_next[i] * x_next[i]
        norm_x = math.sqrt(sq_sum)

        for i in range(n):
            x[i] = x_next[i] / norm_x

        dot = 0.0
        for i in range(n):
            dot += x[i] * v[i]

        c = abs(dot)
        diff = 1.0 - c * c
        if diff < 0.0:
            diff = 0.0
        errors.append(math.sqrt(diff))

    ratios = []
    for i in range(10, len(errors) - 1):
        if errors[i] > 1e-14:
            ratios.append(errors[i + 1] / errors[i])

    ratios_sorted = sorted(ratios)
    m = len(ratios_sorted)
    if m % 2 == 1:
        med = ratios_sorted[m // 2]
    else:
        med = (ratios_sorted[m // 2 - 1] + ratios_sorted[m // 2]) / 2.0

    return float(med)
