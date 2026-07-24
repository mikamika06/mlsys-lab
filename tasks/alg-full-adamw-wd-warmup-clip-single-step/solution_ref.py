import numpy as np


def adamw_single_step(theta, grad, m, v, step, lr, beta1, beta2, eps, weight_decay, warmup_steps, clip_norm):
    theta = np.asarray(theta, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    g = grad.copy()
    norm = np.linalg.norm(g)
    if norm > clip_norm:
        g *= clip_norm / norm

    new_m = beta1 * m + (1.0 - beta1) * g
    new_v = beta2 * v + (1.0 - beta2) * (g * g)

    m_hat = new_m / (1.0 - beta1 ** step)
    v_hat = new_v / (1.0 - beta2 ** step)

    lr_t = lr * min(1.0, step / warmup_steps)

    new_theta = (
        theta
        - lr_t * m_hat / (np.sqrt(v_hat) + eps)
        - lr_t * weight_decay * theta
    )

    return new_theta.astype(np.float64), new_m.astype(np.float64), new_v.astype(np.float64)
