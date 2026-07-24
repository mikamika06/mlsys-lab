import numpy as np

N = 10
N_ITER = 8
_CASES = [(0, 3), (1, 7), (2, 0), (3, 9)]


def _build_case(seed, idx):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    lam = np.arange(1, N + 1, dtype=np.float64) + rng.uniform(-0.1, 0.1, size=N)
    A = (Q * lam) @ Q.T
    A = 0.5 * (A + A.T)

    true_eigs, true_vecs = np.linalg.eigh(A)
    target_eig = float(true_eigs[idx])
    target_vec = true_vecs[:, idx]

    perturb = rng.standard_normal(N) * 0.05
    v0 = target_vec + perturb
    v0 = v0 / np.linalg.norm(v0)
    return A, v0, target_eig


def grade(sol, fx) -> dict:
    max_rel = 0.0
    for seed, idx in _CASES:
        A, v0, target_eig = _build_case(seed, idx)
        try:
            mu = float(sol.rayleigh_quotient_iteration(A.copy(), v0.copy(), N_ITER))
        except Exception:
            return {"rel_err": float("inf")}
        if not np.isfinite(mu):
            return {"rel_err": float("inf")}
        rel = abs(mu - target_eig) / (abs(target_eig) + 1e-12)
        max_rel = max(max_rel, rel)
    return {"rel_err": max_rel}
