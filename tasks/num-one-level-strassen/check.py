import numpy as np


class TraceArray(np.ndarray):
    calls = []

    def __new__(cls, arr):
        return np.asarray(arr, dtype=np.float64).view(cls)

    def __matmul__(self, other):
        TraceArray.calls.append((self.shape, np.asarray(other).shape))
        return np.asarray(self).view(np.ndarray) @ np.asarray(other).view(np.ndarray)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = [
        (4, rng.normal(size=(4, 4)), rng.normal(size=(4, 4))),
        (8, rng.normal(size=(8, 8)), rng.normal(size=(8, 8))),
        (16, rng.normal(size=(16, 16)), rng.normal(size=(16, 16))),
    ]

    worst = 0.0
    counts = []

    for _, A, B in cases:
        ref = np.matmul(A, B)

        TraceArray.calls = []
        try:
            got = np.asarray(
                sol.one_level_strassen(TraceArray(A), TraceArray(B)),
                dtype=np.float64,
            )
        except Exception:
            return {"rel_err": float("inf"), "matmul_count": 0}

        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        worst = max(worst, float(err))
        counts.append(len(TraceArray.calls))

    return {
        "rel_err": worst,
        "matmul_count": 7.0 if all(x == 7 for x in counts) else float(max(counts)),
    }
