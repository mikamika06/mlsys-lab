import numpy as np

def _reference(U, b):
    """Ground-truth back-substitution via NumPy."""
    return np.linalg.solve(U, b)

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(42)
    sizes = [4, 8, 15, 20]
    max_err = 0.0

    for n in sizes:
        U = rng.randn(n, n)
        U = np.triu(U)
        for i in range(n):
            U[i, i] = abs(U[i, i]) + 1.0
        b = rng.randn(n)

        try:
            x_student = sol.back_sub(U.copy(), b.copy())
        except Exception:
            return {"rel_err": 1.0}

        x_student = np.asarray(x_student, dtype=np.float64)
        x_ref = _reference(U, b)
        denom = np.linalg.norm(x_ref) + 1e-12
        err = float(np.linalg.norm(x_student - x_ref) / denom)
        if err > max_err:
            max_err = err

    return {"rel_err": max_err}
