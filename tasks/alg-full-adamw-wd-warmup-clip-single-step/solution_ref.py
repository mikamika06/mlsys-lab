import math
import numpy as np


def adamw_single_step(theta, grad, m, v, step, lr, beta1, beta2, eps, weight_decay, warmup_steps, clip_norm):
    theta = np.asarray(theta, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    g = grad.copy()
    
    sum_sq = 0.0
    for i in range(len(g)):
        sum_sq += g[i] * g[i]
    norm = math.sqrt(sum_sq)

    if norm > clip_norm:
        scale = clip_norm / norm
        for i in range(len(g)):
            g[i] *= scale

    new_m_list = []
    new_v_list = []
    for i in range(len(g)):
        nm = beta1 * m[i] + (1.0 - beta1) * g[i]
        nv = beta2 * v[i] + (1.0 - beta2) * (g[i] * g[i])
        new_m_list.append(nm)
        new_v_list.append(nv)

    new_m = np.array(new_m_list, dtype=np.float64)
    new_v = np.array(new_v_list, dtype=np.float64)

    m_hat_list = []
    v_hat_list = []
    denom1 = 1.0 - (beta1 ** step)
    denom2 = 1.0 - (beta2 ** step)
    for i in range(len(g)):
        m_hat_list.append(new_m[i] / denom1)
        v_hat_list.append(new_v[i] / denom2)

    m_hat = np.array(m_hat_list, dtype=np.float64)
    v_hat = np.array(v_hat_list, dtype=np.float64)

    warmup_scale = min(1.0, step / warmup_steps)
    lr_t = lr * warmup_scale

    new_theta_list = []
    for i in range(len(g)):
        nt = (
            theta[i]
            - lr_t * m_hat[i] / (math.sqrt(v_hat[i]) + eps)
            - lr_t * weight_decay * theta[i]
        )
        new_theta_list.append(nt)

    new_theta = np.array(new_theta_list, dtype=np.float64)

    return new_theta.astype(np.float64), new_m.astype(np.float64), new_v.astype(np.float64)
