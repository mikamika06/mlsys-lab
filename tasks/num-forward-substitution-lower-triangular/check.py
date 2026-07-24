import numpy as np
import scipy.linalg as sla


def _reference(L, b):
    """Ground-truth solve via SciPy's dedicated triangular solver."""
    return sla.solve_triangular(L, b, lower=True)


def grade(sol, fx) -> dict:
    rng = np.random.RandomState(3)
    sizes = [3, 5, 10, 16]
    max_err = 0.0
    shortcut_used = [False]

    orig_np_solve = np.linalg.solve
    orig_sla_solve = sla.solve
    orig_sla_solve_triangular = sla.solve_triangular

    def _flag_np(*a, **k):
        shortcut_used[0] = True
        return orig_np_solve(*a, **k)

    def _flag_sla_solve(*a, **k):
        shortcut_used[0] = True
        return orig_sla_solve(*a, **k)

    def _flag_sla_tri(*a, **k):
        shortcut_used[0] = True
        return orig_sla_solve_triangular(*a, **k)

    try:
        for n in sizes:
            L = np.tril(rng.randn(n, n))
            for i in range(n):
                L[i, i] = abs(L[i, i]) + 1.0
            b = rng.randn(n)
            L_ref, b_ref = L.copy(), b.copy()

            np.linalg.solve = _flag_np
            sla.solve = _flag_sla_solve
            sla.solve_triangular = _flag_sla_tri
            try:
                x = sol.forward_sub(L.copy(), b.copy())
            finally:
                np.linalg.solve = orig_np_solve
                sla.solve = orig_sla_solve
                sla.solve_triangular = orig_sla_solve_triangular

            x = np.asarray(x, dtype=np.float64)
            x_ref = _reference(L_ref, b_ref)
            denom = float(np.linalg.norm(x_ref)) + 1e-12
            err = float(np.linalg.norm(x - x_ref) / denom)
            if err > max_err:
                max_err = err
    except Exception:
        return {"rel_err": 1.0, "no_solver_shortcut": 0.0}

    return {
        "rel_err": max_err,
        "no_solver_shortcut": 0.0 if shortcut_used[0] else 1.0,
    }
