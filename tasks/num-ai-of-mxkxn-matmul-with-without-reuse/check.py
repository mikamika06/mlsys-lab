from mlsys import scorers
import numpy as np

def _ref(M, K, N):
    F = 2.0 * M * N * K
    M_no_reuse = 8.0 * (2 * M * N * K + M * N)
    M_full_reuse = 8.0 * (M * K + K * N + M * N)
    return F / M_no_reuse, F / M_full_reuse

def grade(sol, fx):
    cases = [
        (512, 512, 512),
        (1000, 100, 1000),
        (64, 128, 256),
        (256, 256, 256),
        (1024, 1024, 1024),
    ]
    ref_vals = []
    stu_vals = []
    for M, K, N in cases:
        try:
            stu = sol.ai_matmul(M, K, N)
            stu_vals.extend([float(stu[0]), float(stu[1])])
        except Exception:
            return {"rel_err": 1.0}
        r = _ref(M, K, N)
        ref_vals.extend([r[0], r[1]])
    ref_arr = np.array(ref_vals, dtype=np.float64)
    stu_arr = np.array(stu_vals, dtype=np.float64)
    err = scorers.rel_err(ref_arr, stu_arr)
    return {"rel_err": err}
