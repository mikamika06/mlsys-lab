import numpy as np

def adam_trajectory(
    params0: np.ndarray,
    grads: np.ndarray,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Return the full Adam trajectory with bias correction.

    Parameters
    ----------
    params0 : ndarray, shape (d,)
        Initial parameters.
    grads : ndarray, shape (T, d)
        Sequence of gradients.
    lr : float, optional
        Learning rate.  Default 1e-3.
    beta1 : float, optional
        Exponential decay for the first moment.  Default 0.9.
    beta2 : float, optional
        Exponential decay for the second moment.  Default 0.999.
    eps : float, optional
        Small constant to avoid division by zero.  Default 1e-8.

    Returns
    -------
    ndarray, shape (T+1, d)
        Parameter trajectory: first row is params0, subsequent rows are updated parameters.
    """
    m = np.zeros_like(params0, dtype=np.float64)
    v = np.zeros_like(params0, dtype=np.float64)
    traj = [params0.astype(np.float64).copy()]
    for t, g in enumerate(grads, start=1):
        g = g.astype(np.float64)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        new_params = traj[-1] - lr * m_hat / (np.sqrt(v_hat) + eps)
        traj.append(new_params)
    return np.stack(traj, axis=0)
