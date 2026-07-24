import numpy as np


def jacobian_fd(f, x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(f(x), dtype=np.float64)
    m = y.shape[0]
    n = x.shape[0]
    J = np.empty((m, n), dtype=np.float64)

    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        yp = np.asarray(f(xp), dtype=np.float64)
        ym = np.asarray(f(xm), dtype=np.float64)
        J[:, j] = (yp - ym) / (2.0 * eps)

    return J
