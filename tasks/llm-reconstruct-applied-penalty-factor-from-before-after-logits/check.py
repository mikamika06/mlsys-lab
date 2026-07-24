import numpy as np


def _oracle(before, after, affected_indices):
    x = np.asarray(before, dtype=np.float64)[np.asarray(affected_indices)]
    y = np.asarray(after, dtype=np.float64)[np.asarray(affected_indices)]
    return float(np.exp(np.mean(np.log(x / y))))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1234)
    cases = []

    for size in [4, 8, 16, 32]:
        before = rng.uniform(0.5, 20.0, size=size).astype(np.float64)
        affected = np.arange(0, size, 2, dtype=np.int64)
        factors = rng.uniform(1.2, 4.0, size=len(affected))

        after = before.copy()
        after[affected] = before[affected] / factors
        cases.append((before, after, affected))

    refs = []
    vals = []
    for before, after, affected in cases:
        refs.append(_oracle(before, after, affected))
        try:
            vals.append(float(sol.reconstruct_penalty_factor(before, after, affected)))
        except Exception:
            return {"rel_err": float("inf")}

    ref = np.asarray(refs, dtype=np.float64)
    got = np.asarray(vals, dtype=np.float64)
    err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))
    return {"rel_err": err}
