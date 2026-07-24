import numpy as np


def grade(sol, fx) -> dict:
    cases = [
        np.array([0.1] * 1000, dtype=np.float16),
        np.array([0.1] * 5000, dtype=np.float16),
        np.concatenate(
            [
                np.ones(10000, dtype=np.float16) * np.float16(0.1),
                np.array([1000], dtype=np.float16),
            ]
        ),
        np.concatenate(
            [
                np.ones(20000, dtype=np.float16) * np.float16(0.05),
                np.array([-500], dtype=np.float16),
            ]
        ),
    ]

    worst = 0.0
    for x in cases:
        ref = float(np.sum(x, dtype=np.float64))
        try:
            got = float(sol.kahan_sum_fp16(x.copy()))
        except Exception:
            return {"rel_err": float("inf")}
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, float(err))

    return {"rel_err": worst}
