import math
import numpy as np
from typing import Tuple

def sgd_vs_adam_steps(
    A: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 10000
) -> Tuple[int, int]:
    """
    Return the number of steps required for SGD and Adam to reach a gradient norm < tol.
    Parameters
    ----------
    A : (n, n) symmetric positive‑definite matrix defining f(x)=½xᵀAx
    x0 : initial point; if None defaults to an all‑ones vector
    tol : tolerance on ‖∇f‖₂
    max_iter : maximum number of iterations for each optimizer

    Returns
    -------
    sgd_steps, adam_steps : int
        Number of update steps performed by SGD and Adam respectively.
    """
    n = A.shape[0]
    if x0 is None:
        init = [1.0] * n
    else:
        init = [float(x0[i]) for i in range(n)]

    B = [[float(A[r, c]) for c in range(n)] for r in range(n)]
    for _ in range(100):
        for p in range(n):
            for q in range(p + 1, n):
                apq = B[p][q]
                if math.fabs(apq) < 1e-15:
                    continue
                app = B[p][p]
                aqq = B[q][q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c

                B[p][p] = app - t * apq
                B[q][q] = aqq + t * apq
                B[p][q] = 0.0
                B[q][p] = 0.0

                for i in range(n):
                    if i != p and i != q:
                        a_ip = B[i][p]
                        a_iq = B[i][q]
                        B[i][p] = c * a_ip - s * a_iq
                        B[p][i] = B[i][p]
                        B[i][q] = s * a_ip + c * a_iq
                        B[q][i] = B[i][q]

    L = B[0][0]
    for i in range(1, n):
        if B[i][i] > L:
            L = B[i][i]

    eta = 0.9 / L
    sgd_steps = 0
    x_sgd = list(init)
    for _ in range(max_iter):
        grad = [0.0] * n
        for i in range(n):
            s_val = 0.0
            for j in range(n):
                s_val += float(A[i, j]) * x_sgd[j]
            grad[i] = s_val

        norm_sq = 0.0
        for i in range(n):
            norm_sq += grad[i] * grad[i]
        if math.sqrt(norm_sq) < tol:
            break

        for i in range(n):
            x_sgd[i] -= eta * grad[i]
        sgd_steps += 1

    beta1, beta2, eps, alpha = 0.9, 0.999, 1e-8, 1e-2
    m = [0.0] * n
    v = [0.0] * n
    adam_steps = 0
    x_adam = list(init)
    for t in range(1, max_iter + 1):
        grad = [0.0] * n
        for i in range(n):
            s_val = 0.0
            for j in range(n):
                s_val += float(A[i, j]) * x_adam[j]
            grad[i] = s_val

        norm_sq = 0.0
        for i in range(n):
            norm_sq += grad[i] * grad[i]
        if math.sqrt(norm_sq) < tol:
            break

        b1_t = 1.0 - beta1**t
        b2_t = 1.0 - beta2**t

        for i in range(n):
            g = grad[i]
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)
            m_hat = m[i] / b1_t
            v_hat = v[i] / b2_t
            x_adam[i] -= alpha * m_hat / (math.sqrt(v_hat) + eps)

        adam_steps += 1

    return sgd_steps, adam_steps
