import numpy as np
from scipy.stats import wasserstein_distance


def _cases(rng: np.random.Generator):
    cases = []
    cases.append((np.array([0.0, 1.0, 2.0]), np.array([0.0, 3.0])))
    cases.append((rng.standard_normal(3), rng.standard_normal(1) + 5.0))

    for _ in range(4):
        n = int(rng.integers(2, 60))
        m = int(rng.integers(2, 60))
        u = rng.standard_normal(n) * rng.uniform(0.5, 3.0)
        v = rng.standard_normal(m) * rng.uniform(0.5, 3.0) + rng.uniform(-2.0, 2.0)
        cases.append((u, v))

    return cases


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [(fx["w1_u"], fx["w1_v"])] + _cases(rng)

    max_rel_err = 0.0
    for u, v in cases:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        expected = float(wasserstein_distance(u, v))
        try:
            got = float(sol.wasserstein1_cdf_integral(np.array(u, copy=True), np.array(v, copy=True)))
        except Exception:
            return {"rel_err": float("inf")}

        rel_err = abs(got - expected) / (abs(expected) + 1e-12)
        max_rel_err = max(max_rel_err, rel_err)

    return {"rel_err": max_rel_err}
