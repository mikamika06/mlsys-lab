import numpy as np

from mlsys import probe, scorers

N_ITER = 60
LOOP_BUDGET = float("inf")


def _oracle(A, k):
    """Top-k eigenpairs of a symmetric matrix, straight from NumPy."""
    A = np.asarray(A, dtype=np.float64)
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    order = np.argsort(w)[::-1][:k]
    return w[order], V[:, order]


def _second_case():
    """A second, independently generated problem so the top-k block cannot be memorised."""
    rng = np.random.default_rng(7)
    n, k = 80, 12
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.empty(n, dtype=np.float64)
    lam[:k] = 3.0 - 0.1 * np.arange(k)
    lam[k:] = np.sort(rng.uniform(0.05, 0.30, size=n - k))[::-1]
    A = (U * lam) @ U.T
    A = 0.5 * (A + A.T)
    Q0 = rng.standard_normal((n, k))
    return A, Q0, k


def _call(sol, A, Q0, k):
    out = sol.block_power_topk(A, Q0, N_ITER)
    vals, Q = out[0], out[1]
    vals = np.asarray(vals, dtype=np.float64).ravel()
    Q = np.asarray(Q, dtype=np.float64)
    if vals.shape != (k,) or Q.shape != (A.shape[0], k):
        raise ValueError("wrong output shapes")
    if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(Q)):
        raise ValueError("non-finite output")
    return vals, Q


def _fail():
    return {
        "eigval_rel_err": float("inf"),
        "component_rel_err": float("inf"),
        "subspace_rel_err": float("inf"),
        "line_events": float("inf"),
    }


def grade(sol, fx) -> dict:
    A1 = np.asarray(fx["A"], dtype=np.float64)
    Q01 = np.asarray(fx["Q0"], dtype=np.float64)
    k1 = Q01.shape[1]
    A2, Q02, k2 = _second_case()

    cases = [(A1, Q01, k1), (A2, Q02, k2)]

    val_ref, val_got = [], []
    vec_ref, vec_got = [], []
    proj_ref, proj_got = [], []

    for A, Q0, k in cases:
        w_ref, V_ref = _oracle(A, k)
        try:
            vals, Q = _call(sol, A, Q0, k)
        except Exception:
            return _fail()

        # eigenvector sign is arbitrary -> align before comparing components
        sgn = np.sign(np.sum(Q * V_ref, axis=0))
        sgn[sgn == 0.0] = 1.0
        Q_aligned = Q * sgn

        val_ref.append(w_ref)
        val_got.append(vals)
        vec_ref.append(V_ref.ravel())
        vec_got.append(Q_aligned.ravel())
        proj_ref.append((V_ref @ V_ref.T).ravel())
        proj_got.append((Q @ Q.T).ravel())

    # line-event probe: warm the numpy paths first so lazy imports are not counted
    try:
        sol.block_power_topk(A1, Q01, N_ITER)
        events = probe.count_line_events(sol.block_power_topk, A1, Q01, N_ITER)
    except Exception:
        return _fail()

    return {
        "eigval_rel_err": scorers.rel_err(np.concatenate(val_ref), np.concatenate(val_got)),
        "component_rel_err": scorers.rel_err(np.concatenate(vec_ref), np.concatenate(vec_got)),
        "subspace_rel_err": scorers.rel_err(np.concatenate(proj_ref), np.concatenate(proj_got)),
        "line_events": float(events),
    }
