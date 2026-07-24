import numpy as np

def _standard_adam(params, grads, lr, beta1, beta2, eps):
    """Unsharded Adam — the correctness oracle."""
    params = np.asarray(params, dtype=np.float64).copy()
    grads = np.asarray(grads, dtype=np.float64)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    T = len(grads)
    for t in range(1, T + 1):
        g = grads[t - 1]
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params

def grade(sol, fx) -> dict:
    rng = np.random.RandomState(2024)
    cases = [
        (20, 4, 1),        # num_ranks = 1 (trivial)
        (20, 1, 2),        # single step
        (20, 5, 2),        # even split
        (100, 10, 8),      # larger N, many ranks
        (37, 3, 5),        # non-divisible N / R
        (64, 20, 16),      # N = R * base, many steps
        (128, 8, 3),       # remainder = 2
    ]
    max_err = 0.0
    for N, T, R in cases:
        params = rng.randn(N)
        grads = rng.randn(T, N)
        try:
            got = np.asarray(
                sol.zero_one_adam(params.copy(), grads, R),
                dtype=np.float64,
            )
            ref = _standard_adam(params, grads, 0.001, 0.9, 0.999, 1e-8)
            err = float(np.max(np.abs(got - ref)))
            max_err = max(max_err, err)
        except Exception:
            return {"max_abs_err": float("inf")}
    return {"max_abs_err": max_err}
