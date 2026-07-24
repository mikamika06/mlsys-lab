import numpy as np

def _reference(A, b, init_x, lr, momentum, T):
    x = init_x.astype(np.float64)
    v = np.zeros_like(x)
    for _ in range(T):
        grad = A @ x - b
        v = momentum * v - lr * grad
        x = x + v
    return x

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0
    for _ in range(5):
        n = rng.integers(3, 8)
        M = rng.standard_normal((n, n))
        A = M.T @ M + np.eye(n) * 1e-3  # SPD
        b = rng.standard_normal(n)
        init_x = rng.standard_normal(n)
        lr = 0.01 * (rng.random() + 0.5)   # between .005 and .015
        momentum = rng.random() * 0.9      # [0,0.9)
        T = rng.integers(10, 50)

        try:
            cand = sol.sgd_momentum_quadratic(A, b, init_x, lr, momentum, T)
        except Exception:
            return {"rel_err": float("inf")}

        ref = _reference(A, b, init_x, lr, momentum, T)
        cand = np.asarray(cand, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)

        err = np.linalg.norm(cand - ref) / (np.linalg.norm(ref) + 1e-12)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
