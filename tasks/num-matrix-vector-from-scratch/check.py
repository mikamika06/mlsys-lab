import numpy as np

from mlsys import probe, scorers

_CASES = [
    (5, 7),
    (8, 4),
    (3, 9),
]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    total_ops = 0

    for n, m in _CASES:
        A = rng.standard_normal((n, m)).astype(np.float64)
        x = rng.standard_normal(m).astype(np.float64)
        ref = A @ x

        try:
            ops = probe.count_line_events(sol.matvec_from_scratch, A, x)
            got = sol.matvec_from_scratch(A, x)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "op_count": 0.0}

        total_ops += ops
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf"), "op_count": float(total_ops)}
        max_err = max(max_err, scorers.max_abs_err(ref, got))

    return {"max_abs_err": float(max_err), "op_count": float(total_ops)}
