import numpy as np


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = []

    for n in [8, 10]:
        X = rng.normal(size=(n, n))
        A = (X + X.T) / 2.0
        A += np.diag(np.linspace(1.0, 3.0, n))
        cases.append(A)

    max_err = 0.0
    for A in cases:
        ref = np.linalg.eigvalsh(A)
        try:
            got = np.asarray(sol.wilkinson_eigvals(A.copy(), 50), dtype=np.float64)
        except Exception:
            return {"rel_err": 1.0}

        if got.shape != ref.shape:
            return {"rel_err": 1.0}

        got = np.sort(got)
        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        max_err = max(max_err, float(err))

    return {"rel_err": max_err}
