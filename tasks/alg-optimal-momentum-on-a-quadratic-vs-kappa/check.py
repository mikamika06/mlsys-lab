import numpy as np

def _reference_beta(A):
    eigs = np.linalg.eigvalsh(A)
    kappa = eigs[-1] / eigs[0]
    return ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_rel_err = 0.0
    for n in [3, 5, 10]:
        for _ in range(5):
            M = rng.standard_normal((n, n))
            A = M @ M.T + n * np.eye(n)  # ensure SPD
            try:
                got = sol.optimal_momentum_beta(A)
            except Exception:
                return {"rel_err": float("inf")}
            ref = _reference_beta(A)
            rel_err = abs(got - ref) / (abs(ref) + 1e-12)
            if rel_err > max_rel_err:
                max_rel_err = rel_err
    return {"rel_err": max_rel_err}
