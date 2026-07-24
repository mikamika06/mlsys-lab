import numpy as np

def _run_sgd(A, x0, tol, max_iter):
    L = np.linalg.eigvalsh(A).max()
    eta = 0.9 / L
    x = x0.copy()
    for t in range(1, max_iter + 1):
        grad = A @ x
        if np.linalg.norm(grad) < tol:
            return t - 1
        x -= eta * grad
    return max_iter

def _run_adam(A, x0, tol, max_iter):
    beta1, beta2, eps, alpha = 0.9, 0.999, 1e-8, 1e-2
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    x = x0.copy()
    for t in range(1, max_iter + 1):
        grad = A @ x
        if np.linalg.norm(grad) < tol:
            return t - 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= alpha * m_hat / (np.sqrt(v_hat) + eps)
    return max_iter

def _reference(A, x0, tol, max_iter):
    if x0 is None:
        x0 = np.ones(A.shape[0], dtype=np.float64)
    else:
        x0 = np.asarray(x0, dtype=np.float64)
    sgd_steps = _run_sgd(A, x0, tol, max_iter)
    adam_steps = _run_adam(A, x0, tol, max_iter)
    return (sgd_steps, adam_steps)

def grade(sol, fx) -> dict:
    # deterministic test cases
    np.random.seed(0)
    tests = []

    # 2x2 ill‑conditioned
    A1 = np.diag([1., 1000.])
    tests.append((A1, None))

    # 5x5 diagonal with increasing eigenvalues
    eigs = np.arange(1, 6) * 200.
    A2 = np.diag(eigs)
    tests.append((A2, None))

    # identity matrix (well‑conditioned)
    A3 = np.eye(4)
    tests.append((A3, None))

    # random SPD with moderate condition number
    R = np.random.randn(6, 6)
    A4 = R.T @ R + 0.1 * np.eye(6)
    eigs4 = np.linalg.eigvalsh(A4)
    scale = 100 / eigs4.max()
    A4 *= scale
    tests.append((A4, None))

    # 20x20 diagonal with large condition number
    eigs5 = np.arange(1, 21) * 10.
    A5 = np.diag(eigs5)
    tests.append((A5, None))

    ok = 1.0
    for A, x0 in tests:
        try:
            ref = _reference(A, x0, tol=1e-6, max_iter=20000)
            got = sol.sgd_vs_adam_steps(A, x0, tol=1e-6, max_iter=20000)
        except Exception:
            ok = 0.0
            break
        if ref != tuple(got):
            ok = 0.0
            break

    return {"exact_match": ok}
