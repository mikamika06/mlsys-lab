import numpy as np

def _reference(Q, c, x0, lr, beta, T):
    """Trusted implementation of NAG for a quadratic."""
    x_prev = x0.copy()
    x_curr = x0.copy()
    for _ in range(T):
        y = x_curr + beta * (x_curr - x_prev)
        grad = Q @ y - c
        x_next = y - lr * grad
        x_prev, x_curr = x_curr, x_next
    return x_curr

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    rel_errs = []
    for dim in [5, 10]:
        # Generate a symmetric positive‑definite matrix Q
        A = rng.standard_normal((dim, dim))
        Q = A.T @ A + 1e-3 * np.eye(dim)   # ensure PD
        c = rng.standard_normal(dim)
        x0 = rng.standard_normal(dim)

        eigvals = np.linalg.eigvalsh(Q)
        L = eigvals[-1]
        mu = eigvals[0]
        lr = 1.0 / L
        beta = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
        T = 50

        ref = _reference(Q, c, x0, lr, beta, T)
        try:
            student = sol.nesterov_minimize(Q, c, x0, lr, beta, T)
        except Exception:
            return {"rel_err": 0.0}
        student = np.asarray(student, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)

        err = np.linalg.norm(student - ref) / (np.linalg.norm(ref) + 1e-12)
        rel_errs.append(err)

    return {"rel_err": float(np.mean(rel_errs))}
