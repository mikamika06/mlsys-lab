import numpy as np


def _principal_angle(U, U_ref):
    qa, _ = np.linalg.qr(U)
    qb, _ = np.linalg.qr(U_ref)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    singular = np.clip(singular, -1.0, 1.0)
    return float(np.arccos(np.min(singular)))


def _rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1234)
    m, n, k = 160, 100, 5

    base = rng.normal(size=(m, k)) @ rng.normal(size=(k, n))
    noise = 0.01 * rng.normal(size=(m, n))
    A = base + noise
    seed = 42

    try:
        U, S, Vt = sol.randomized_svd(A, k, seed)
    except Exception:
        return {"rel_err": 1.0, "subspace_angle": float(np.pi)}

    U = np.asarray(U, dtype=float)
    S = np.asarray(S, dtype=float)
    Vt = np.asarray(Vt, dtype=float)

    if U.shape != (m, k) or S.shape != (k,) or Vt.shape != (k, n):
        return {"rel_err": 1.0, "subspace_angle": float(np.pi)}

    _, s_ref, _ = np.linalg.svd(A, full_matrices=False)
    u_ref, _, _ = np.linalg.svd(A, full_matrices=False)

    return {
        "rel_err": _rel_err(S, s_ref[:k]),
        "subspace_angle": _principal_angle(U, u_ref[:, :k]),
    }
