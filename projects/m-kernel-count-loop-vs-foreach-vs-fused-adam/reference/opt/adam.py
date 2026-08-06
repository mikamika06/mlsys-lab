import numpy as np
from opt.grouping import group_tensors_by_device_dtype


def step_loop(params, states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    for p_info, state in zip(params, states):
        p = p_info["param"]
        g = p_info["grad"]

        state["step"] += 1
        t = state["step"]

        if weight_decay != 0.0:
            g = g + weight_decay * p

        m = state["exp_avg"]
        v = state["exp_avg_sq"]

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)

        state["exp_avg"] = m
        state["exp_avg_sq"] = v

        bias_correction1 = 1.0 - (beta1 ** t)
        bias_correction2 = 1.0 - (beta2 ** t)

        step_size = lr * (np.sqrt(bias_correction2) / bias_correction1)
        p -= step_size * (m / (np.sqrt(v) + eps))


def step_foreach(params, states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    grouped = group_tensors_by_device_dtype(params)
    param_to_state = {id(p_info["param"]): st for p_info, st in zip(params, states)}

    for key, group in grouped.items():
        if not group:
            continue

        for p_info in group:
            p = p_info["param"]
            g = p_info["grad"]
            st = param_to_state[id(p)]

            st["step"] += 1
            t = st["step"]

            if weight_decay != 0.0:
                g = g + weight_decay * p

            m = st["exp_avg"]
            v = st["exp_avg_sq"]

            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g ** 2)

            st["exp_avg"] = m
            st["exp_avg_sq"] = v

            bias_correction1 = 1.0 - (beta1 ** t)
            bias_correction2 = 1.0 - (beta2 ** t)

            step_size = lr * (np.sqrt(bias_correction2) / bias_correction1)
            p -= step_size * (m / (np.sqrt(v) + eps))


def step_fused(params, states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    for p_info, st in zip(params, states):
        p = p_info["param"]
        g = p_info["grad"]

        st["step"] += 1
        t = st["step"]

        p_flat = p.ravel()
        g_flat = g.ravel()
        m_flat = st["exp_avg"].ravel()
        v_flat = st["exp_avg_sq"].ravel()

        for i in range(p_flat.size):
            gi = g_flat[i]
            if weight_decay != 0.0:
                gi = gi + weight_decay * p_flat[i]

            mi = beta1 * m_flat[i] + (1.0 - beta1) * gi
            vi = beta2 * v_flat[i] + (1.0 - beta2) * (gi * gi)

            m_flat[i] = mi
            v_flat[i] = vi

            bc1 = 1.0 - (beta1 ** t)
            bc2 = 1.0 - (beta2 ** t)

            step_size = lr * (np.sqrt(bc2) / bc1)
            p_flat[i] -= step_size * (mi / (np.sqrt(vi) + eps))
