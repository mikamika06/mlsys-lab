import numpy as np


def _oracle(a):
    return float(np.sum(a.astype(np.float64), dtype=np.float64))


def grade(sol, fx) -> dict:
    cases = [
        np.array([10000, 1, -10000, 1, 1], dtype=np.float16),
        np.concatenate(
            [
                np.full(2000, 0.5, dtype=np.float16),
                np.full(2000, -0.5, dtype=np.float16),
                np.ones(2000, dtype=np.float16),
            ]
        ),
        np.array([10000] + [0.25] * 4000 + [-10000], dtype=np.float16),
        (np.arange(1, 5001, dtype=np.float32) % 7).astype(np.float16) - np.float16(2.5),
    ]

    worst = 0.0
    for a in cases:
        ref = _oracle(a)
        try:
            got = float(sol.kahan_sum_fp16(a))
        except Exception:
            return {"rel_err": float("inf")}
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)
    return {"rel_err": worst}
