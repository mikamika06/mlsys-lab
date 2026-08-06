import math
import numpy as np


def offloaded_adamw_step(
    param,
    grad,
    m,
    v,
    step,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
):
    cpu_param = np.asarray(param, dtype=np.float64).copy()
    cpu_grad = np.asarray(grad, dtype=np.float64).copy()
    cpu_m = np.asarray(m, dtype=np.float64).copy()
    cpu_v = np.asarray(v, dtype=np.float64).copy()

    shape = cpu_param.shape
    flat_param = cpu_param.ravel()
    flat_grad = cpu_grad.ravel()
    flat_m = cpu_m.ravel()
    flat_v = cpu_v.ravel()

    new_m = np.empty_like(flat_m)
    new_v = np.empty_like(flat_v)
    new_param = np.empty_like(flat_param)

    b1_pow = beta1 ** step
    b2_pow = beta2 ** step

    for i in range(flat_param.shape[0]):
        g = flat_grad[i]
        p = flat_param[i]
        
        m_val = beta1 * flat_m[i] + (1.0 - beta1) * g
        v_val = beta2 * flat_v[i] + (1.0 - beta2) * (g * g)
        
        new_m[i] = m_val
        new_v[i] = v_val
        
        m_hat = m_val / (1.0 - b1_pow)
        v_hat = v_val / (1.0 - b2_pow)
        
        p_val = p - lr * (m_hat / (math.sqrt(v_hat) + eps) + weight_decay * p)
        new_param[i] = p_val

    return new_param.reshape(shape), new_m.reshape(shape), new_v.reshape(shape)
