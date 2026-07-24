import numpy as np


def _oracle(A):
    return np.sum(A, axis=1, dtype=np.float64).astype(np.float32)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = [
        np.array(
            [
                [1000, 1, -1000, 1],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float16,
        ),
        rng.normal(0, 1, size=(32, 4096)).astype(np.float16),
        (
            np.ones((8, 8192), dtype=np.float16)
            * np.float16(0.25)
        ),
    ]

    worst = 0.0
    for A in cases:
        try:
            got = sol.sum_rows_fp32(A)
        except Exception:
            return {"rel_err": 1.0}

        ref = _oracle(A)
        worst = max(worst, _rel_err(got, ref))

    return {"rel_err": worst}
