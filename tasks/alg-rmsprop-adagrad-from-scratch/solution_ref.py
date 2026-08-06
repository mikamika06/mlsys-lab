import math
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
    traj = np.zeros((T + 1, d), dtype=np.float64)
    v = [0.0] * d
    theta = [0.0] * d
    for t in range(T):
        for j in range(d):
            g = grads[t, j]
            v[j] = decay_rate * v[j] + (1.0 - decay_rate) * (g * g)
            theta[j] -= lr * g / (math.sqrt(v[j]) + eps)
            traj[t + 1, j] = theta[j]
    return traj
