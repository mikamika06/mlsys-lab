import numpy as np


def _oracle(x):
    return float(np.sum(x, dtype=np.float64))


def grade(sol, fx) -> dict:
    cases = []

    rng = np.random.default_rng(12345)

    a = np.empty(20000, dtype=np.float16)
    a[0::2] = np.float16(10000.0)
    a[1::2] = np.float16(-10000.0)
    a[::4] += np.float16(0.5)
    cases.append(a)

    b = rng.normal(0, 1, size=30000).astype(np.float16)
    b[:10000] += np.float16(5000)
    b[10000:20000] -= np.float16(5000)
    cases.append(b)

    c = np.full(40000, np.float16(0.25), dtype=np.float16)
    c[::3] = np.float16(-0.25)
    cases.append(c)

    worst = 0.0
    for x in cases:
        try:
            got = float(sol.compensated_sum(x))
        except Exception:
            return {"rel_err": float("inf")}

        ref = _oracle(x)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, float(err))

    return {"rel_err": worst}
