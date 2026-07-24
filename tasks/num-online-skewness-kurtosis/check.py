import numpy as np


def _reference(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    mean = np.mean(x)
    centered = x - mean
    m2 = np.sum(centered ** 2)
    if m2 == 0.0:
        return np.array([0.0, 0.0], dtype=np.float64)
    m3 = np.sum(centered ** 3)
    m4 = np.sum(centered ** 4)
    n = float(x.size)
    var = m2 / n
    skew = (m3 / n) / (var ** 1.5)
    kurt = (m4 / n) / (var ** 2) - 3.0
    return np.array([skew, kurt], dtype=np.float64)


def _rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        rng.normal(size=128),
        rng.normal(loc=100000.0, scale=3.0, size=257),
        np.linspace(-5, 5, 101),
        rng.standard_t(df=5, size=333),
        np.array([7.0] * 20),
    ]

    worst = 0.0
    for values in cases:
        try:
            got = np.asarray(sol.online_moments(values.tolist()), dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}
        ref = _reference(values)
        if got.shape != (2,):
            return {"rel_err": 1.0}
        worst = max(worst, _rel_err(got, ref))
    return {"rel_err": worst}
