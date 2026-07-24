import numpy as np

def rmsprop_trajectory(
    grads: np.ndarray,
    lr: float = 0.01,
    eps: float = 1e-8,
    decay_rate: float = 0.9
) -> np.ndarray:
    """
    Compute the RMSProp trajectory for a sequence of gradients.

    Parameters
    ----------
    grads : (T, d) array_like
        Sequence of gradient vectors; each row is g_t.
    lr : float, optional
        Learning rate η.
    eps : float, optional
        Small constant added to denominator.
    decay_rate : float, optional
        Decay rate ρ for the squared‑gradient accumulator.

    Returns
    -------
    trajectory : (T+1, d) ndarray of dtype float64
        Parameter vectors θ_0 … θ_T.  The initial vector is all zeros.
    """
    grads = np.asarray(grads, dtype=np.float64)
    T, d = grads.shape
    theta = np.zeros(d, dtype=np.float64)
    v = np.zeros(d, dtype=np.float64)
    traj = [theta.copy()]
    for g in grads:
        v = decay_rate * v + (1 - decay_rate) * g**2
        theta -= lr * g / (np.sqrt(v) + eps)
        traj.append(theta.copy())
    return np.stack(traj, axis=0)
