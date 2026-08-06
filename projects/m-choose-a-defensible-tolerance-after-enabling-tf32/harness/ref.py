import numpy as np


def get_test_cases():
    rng = np.random.default_rng(42)
    cases = []
    for _ in range(5):
        shape = (64, 64)
        a = rng.standard_normal(shape).astype(np.float32)
        b = a + rng.standard_normal(shape).astype(np.float32) * 1e-4
        cases.append((a, b))
    return cases


def reference_error(a, b):
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diff = np.abs(arr_a - arr_b)
    denom = np.maximum(np.abs(arr_b), 1e-12)
    return float(np.max(diff / denom))


def reference_tolerance(shape, condition_number):
    m, n = shape
    k = max(m, n)
    eps_tf32 = 2.0 ** -10
    cond = max(float(condition_number), 1.0)
    tol = float(k * eps_tf32 * cond * 2.0)
    return max(tol, 1e-4)
