import numpy as np


def suggest_tolerance(shape, condition_number):
    m, n = shape
    k = max(m, n)
    eps_tf32 = 2.0 ** -10
    cond = max(float(condition_number), 1.0)
    tol = float(k * eps_tf32 * cond * 2.0)
    return max(tol, 1e-4)
