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

    cpu_m = beta1 * cpu_m + (1.0 - beta1) * cpu_grad
    cpu_v = beta2 * cpu_v + (1.0 - beta2) * (cpu_grad * cpu_grad)

    m_hat = cpu_m / (1.0 - beta1 ** step)
    v_hat = cpu_v / (1.0 - beta2 ** step)

    cpu_param = cpu_param - lr * (
        m_hat / (np.sqrt(v_hat) + eps) + weight_decay * cpu_param
    )

    return cpu_param, cpu_m, cpu_v
