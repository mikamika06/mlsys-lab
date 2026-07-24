import numpy as np

def _uld_loss(s, t):
    """Unsorted Label Distance loss."""
    return np.sum((np.sort(s) - np.sort(t)) ** 2)

def grade(sol, fx) -> dict:
    np.random.seed(2024)  # fixed seed for reproducibility

    test_cases = []
    for _ in range(6):
        n = np.random.randint(4, 8)
        s = np.random.randn(n).astype(np.float64)
        t = np.random.randn(n).astype(np.float64)
        test_cases.append((s, t))

    max_abs_err = 0.0
    eps = 1e-5

    for s, t in test_cases:
        try:
            g_analytic = sol.uld_gradient(s, t)
        except Exception:
            return {"max_abs_err": float("inf")}

        # finite-difference oracle
        g_fd = np.zeros_like(s)
        for i in range(len(s)):
            s_plus = s.copy()
            s_plus[i] += eps
            s_minus = s.copy()
            s_minus[i] -= eps
            g_fd[i] = (_uld_loss(s_plus, t) - _uld_loss(s_minus, t)) / (2 * eps)

        err = np.max(np.abs(g_analytic - g_fd))
        if err > max_abs_err:
            max_abs_err = err

    return {"max_abs_err": max_abs_err}
