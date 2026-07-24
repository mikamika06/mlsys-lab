import numpy as np

def _oracle(K: np.ndarray) -> int:
    # Compute variance of per‑row ranges
    row_ranges = np.ptp(K, axis=1)
    col_ranges = np.ptp(K, axis=0)
    var_rows = np.var(row_ranges)
    var_cols = np.var(col_ranges)
    return 0 if var_rows > var_cols else 1

def grade(sol, fx) -> dict:
    # Test cases: deterministic arrays
    tests = [
        np.array([[0, 1, 2], [3, 4, 5]]),          # equal variances → axis 0
        np.array([[10, -10], [20, -20], [30, -30]]),# rows have larger spread
        np.random.default_rng(42).integers(-100, 101, size=(4,3)),
        np.random.default_rng(12345).random((5,2))*200-100,
        np.array([[0,0,0],[1,1,1],[2,2,2]]),       # columns have larger spread
    ]
    for K in tests:
        try:
            got = sol.classify_quant_axis(K)
        except Exception:
            return {"argmin_index": 0.0}
        ref = _oracle(K)
        if int(got) != ref:
            return {"argmin_index": 0.0}
    return {"argmin_index": 1.0}
