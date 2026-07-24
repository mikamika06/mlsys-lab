import numpy as np


def estimate_convergence_rate(A):
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    x0 = np.arange(1, n + 1, dtype=np.float64)
    x0 /= np.linalg.norm(x0)

    v = x0.copy()
    for _ in range(50):
        v = A @ v
        v /= np.linalg.norm(v)

    x = x0.copy()
    errors = []

    for _ in range(50):
        x = A @ x
        x /= np.linalg.norm(x)
        c = abs(float(np.dot(x, v)))
        errors.append(float(np.sqrt(max(0.0, 1.0 - c * c))))

    ratios = []
    for i in range(10, len(errors) - 1):
        if errors[i] > 1e-14:
            ratios.append(errors[i + 1] / errors[i])

    return float(np.median(ratios))
