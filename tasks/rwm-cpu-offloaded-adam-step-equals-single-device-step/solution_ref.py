import math


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
    new_param = []
    new_m = []
    new_v = []

    b1_pow = beta1 ** step
    b2_pow = beta2 ** step

    for i in range(len(param)):
        g = grad[i]
        p = param[i]

        m_val = beta1 * m[i] + (1.0 - beta1) * g
        v_val = beta2 * v[i] + (1.0 - beta2) * (g * g)

        new_m.append(m_val)
        new_v.append(v_val)

        m_hat = m_val / (1.0 - b1_pow)
        v_hat = v_val / (1.0 - b2_pow)

        p_val = p - lr * (m_hat / (math.sqrt(v_hat) + eps) + weight_decay * p)
        new_param.append(p_val)

    return new_param, new_m, new_v
