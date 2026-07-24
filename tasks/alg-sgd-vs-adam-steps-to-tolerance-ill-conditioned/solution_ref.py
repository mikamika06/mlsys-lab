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
    if x0 is None:
        init = np.ones(A.shape[0], dtype=np.float64)
    else:
        init = np.asarray(x0, dtype=np.float64)

    # ---------- SGD ----------
    L = np.linalg.eigvalsh(A).max()
    eta = 0.9 / L
    sgd_steps = 0
    x_sgd = init.copy()
    for _ in range(max_iter):
        grad = A @ x_sgd
        if np.linalg.norm(grad) < tol:
            break
        x_sgd -= eta * grad
        sgd_steps += 1

    # ---------- Adam ----------
    beta1, beta2, eps, alpha = 0.9, 0.999, 1e-8, 1e-2
    m = np.zeros_like(init)
    v = np.zeros_like(init)
    adam_steps = 0
    x_adam = init.copy()
    for t in range(1, max_iter + 1):
        grad = A @ x_adam
        if np.linalg.norm(grad) < tol:
            break
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x_adam -= alpha * m_hat / (np.sqrt(v_hat) + eps)
        adam_steps += 1

    return sgd_steps, adam_steps
