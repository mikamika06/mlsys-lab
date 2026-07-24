import numpy as np

def _reference(m, k, n, dtype):
    a = np.zeros((m, k), dtype=dtype)
    b = np.zeros((k, n), dtype=dtype)
    bytes_read = a.nbytes + b.nbytes
    c_bytes = (m * n) * a.itemsize
    total_bytes = bytes_read + c_bytes
    flops = 2 * m * k * n
    return flops / total_bytes

def grade(sol, fx) -> dict:
    cases = [
        ((10, 20, 30), np.float32),
        ((5, 5, 5), np.float64),
        ((3, 7, 11), np.int8),
    ]
    max_rel_err = 0.0
    for (m, k, n), dtype in cases:
        try:
            got = sol.arithmetic_intensity(m, k, n, dtype)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _reference(m, k, n, dtype)
        rel = abs(got - ref) / (abs(ref) + 1e-12)
        if rel > max_rel_err:
            max_rel_err = rel
    return {"rel_err": max_rel_err}
