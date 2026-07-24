import numpy as np

from mlsys import probe, scorers

_CASES = [
    (6, 5, 7),
    (4, 8, 3),
    (9, 3, 6),
]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    total_ops = 0

    for n, k, m in _CASES:
        A = rng.standard_normal((n, k)).astype(np.float64)
        B = rng.standard_normal((k, m)).astype(np.float64)
        ref = A @ B

        try:
            ops = probe.count_line_events(sol.matmul_ikj, A, B)
            got = sol.matmul_ikj(A, B)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "op_count": 0.0}

        total_ops += ops
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf"), "op_count": float(total_ops)}
        max_err = max(max_err, scorers.max_abs_err(ref, got))

    return {"max_abs_err": float(max_err), "op_count": float(total_ops)}
