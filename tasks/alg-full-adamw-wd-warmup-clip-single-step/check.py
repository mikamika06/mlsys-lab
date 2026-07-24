import numpy as np


def _adamw_oracle(theta, grad, m, v, step, lr, beta1, beta2, eps, weight_decay, warmup_steps, clip_norm):
    theta = np.asarray(theta, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    g = grad.copy()
    norm = np.linalg.norm(g)
    if norm > clip_norm:
        g = g * (clip_norm / norm)

    new_m = beta1 * m + (1.0 - beta1) * g
    new_v = beta2 * v + (1.0 - beta2) * (g * g)

    m_hat = new_m / (1.0 - beta1 ** step)
    v_hat = new_v / (1.0 - beta2 ** step)

    warmup_scale = min(1.0, step / warmup_steps)
    lr_t = lr * warmup_scale

    new_theta = (
        theta
        - lr_t * m_hat / (np.sqrt(v_hat) + eps)
        - lr_t * weight_decay * theta
    )
    return new_theta, new_m, new_v


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.0, -2.0, 0.5]),
            np.array([0.5, -0.5, 0.25]),
            np.zeros(3),
            np.zeros(3),
            1, 0.001, 0.9, 0.999, 1e-8, 0.01, 10, 1.0,
        ),
        (
            np.array([3.0, -1.0]),
            np.array([10.0, -10.0]),
            np.array([0.2, -0.1]),
            np.array([0.03, 0.04]),
            5, 0.002, 0.85, 0.995, 1e-6, 0.02, 3, 0.5,
        ),
        (
            np.array([-0.2, 0.4, 1.5, -3.0]),
            np.array([0.01, -0.02, 0.03, -0.04]),
            np.array([0.1, 0.1, -0.1, 0.0]),
            np.array([0.01, 0.02, 0.03, 0.04]),
            20, 0.01, 0.9, 0.999, 1e-8, 0.0, 5, 10.0,
        ),
    ]

    worst = 0.0
    for case in cases:
        try:
            got = sol.adamw_single_step(*case)
            ref = _adamw_oracle(*case)
            for a, b in zip(got, ref):
                worst = max(worst, float(np.max(np.abs(np.asarray(a) - b))))
        except Exception:
            return {"max_abs_err": float("inf")}
    return {"max_abs_err": worst}
