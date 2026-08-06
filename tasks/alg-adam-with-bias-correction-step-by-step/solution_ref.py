import math

def adam_trajectory(
    params0: list[float],
    grads: list[list[float]],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> list[list[float]]:
    """
    Return the full Adam trajectory with bias correction.

    Parameters
    ----------
    params0 : list[float]
        Initial parameters.
    grads : list[list[float]]
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
    list[list[float]]
        Parameter trajectory: first row is params0, subsequent rows are updated parameters.
    """
    d = len(params0)
    T = len(grads)
    out = [[0.0] * d for _ in range(T + 1)]
    m = [0.0] * d
    v = [0.0] * d

    current_params = [float(params0[i]) for i in range(d)]
    for i in range(d):
        out[0][i] = current_params[i]

    for t in range(1, T + 1):
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        for i in range(d):
            g = float(grads[t - 1][i])
            m[i] = beta1 * m[i] + (1.0 - beta1) * g
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)
            m_hat = m[i] / bc1
            v_hat = v[i] / bc2
            current_params[i] = current_params[i] - lr * m_hat / (math.sqrt(v_hat) + eps)
            out[t][i] = current_params[i]

    return out
