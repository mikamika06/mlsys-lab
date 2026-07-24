import numpy as np

from mlsys import scorers


def _cases(fx):
    """Fixture SPD matrix plus two matrices the grader builds itself."""
    A1 = np.asarray(fx["A"], dtype=np.float64)

    rng = np.random.default_rng(3)
    n = 30
    B = rng.standard_normal((n, n))
    A2 = (B @ B.T) / n + 1.5 * np.eye(n)
    A2 = 0.5 * (A2 + A2.T)

    A3 = np.array([
        [4.0, 2.0, -2.0],
        [2.0, 10.0, 2.0],
        [-2.0, 2.0, 5.0],
    ])
    return [A1, A2, A3]


def _fail():
    return {
        "recon_max_abs_err": float("inf"),
        "factor_max_abs_err": float("inf"),
        "upper_violation": float("inf"),
        "min_diag": -1.0,
        "builtin_used": 1.0,
    }


def grade(sol, fx) -> dict:
    cases = _cases(fx)

    # ---- oracle first, with the real NumPy routine, before any patching ----
    refs = [np.linalg.cholesky(A) for A in cases]

    used = {"flag": False}
    patched = []

    def _spy(orig):
        def wrapper(*a, **k):
            used["flag"] = True
            return orig(*a, **k)
        return wrapper

    # Forbid the library shortcut: any call to a built-in Cholesky is recorded.
    targets = [(np.linalg, "cholesky")]
    try:  # scipy is optional; guard it if present
        import scipy.linalg as sla  # noqa: F401
        targets += [(sla, "cholesky"), (sla, "cho_factor")]
    except Exception:
        pass
    for mod, name in targets:
        orig = getattr(mod, name, None)
        if orig is None:
            continue
        patched.append((mod, name, orig))
        setattr(mod, name, _spy(orig))

    outs = []
    try:
        for A in cases:
            L = sol.cholesky_lower(A)
            L = np.asarray(L, dtype=np.float64)
            if L.shape != A.shape or not np.all(np.isfinite(L)):
                return _fail()
            outs.append(L)
    except Exception:
        return _fail()
    finally:
        for mod, name, orig in patched:
            setattr(mod, name, orig)

    recon_err = 0.0
    factor_err = 0.0
    upper = 0.0
    min_diag = float("inf")
    for A, L, Lref in zip(cases, outs, refs):
        recon_err = max(recon_err, scorers.max_abs_err(A, L @ L.T))
        factor_err = max(factor_err, scorers.max_abs_err(Lref, L))
        upper = max(upper, float(np.max(np.abs(np.triu(L, 1)))))
        min_diag = min(min_diag, float(np.min(np.diag(L))))

    return {
        "recon_max_abs_err": float(recon_err),
        "factor_max_abs_err": float(factor_err),
        "upper_violation": float(upper),
        "min_diag": float(min_diag),
        "builtin_used": 1.0 if used["flag"] else 0.0,
    }
