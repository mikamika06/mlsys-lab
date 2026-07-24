import numpy as np


def _adamw_oracle(param, grad, m, v, step, lr, beta1, beta2, eps, weight_decay):
    p = np.asarray(param, dtype=np.float64).copy()
    g = np.asarray(grad, dtype=np.float64).copy()
    m_new = np.asarray(m, dtype=np.float64).copy()
    v_new = np.asarray(v, dtype=np.float64).copy()

    m_new = beta1 * m_new + (1.0 - beta1) * g
    v_new = beta2 * v_new + (1.0 - beta2) * (g * g)

    m_hat = m_new / (1.0 - beta1 ** step)
    v_hat = v_new / (1.0 - beta2 ** step)

    p = p - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * p)
    return p, m_new, v_new


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([1.0, -2.0, 0.5], dtype=np.float64),
            np.array([0.1, -0.2, 0.3], dtype=np.float64),
            np.zeros(3),
            np.zeros(3),
            1,
            1e-2,
            0.9,
            0.999,
            1e-8,
            0.01,
        ),
        (
            np.array([3.0, -1.5, 2.25, 0.0]),
            np.array([-0.4, 0.2, 0.1, -0.7]),
            np.array([0.05, -0.01, 0.02, 0.0]),
            np.array([0.002, 0.004, 0.001, 0.0]),
            7,
            5e-4,
            0.85,
            0.995,
            1e-7,
            0.02,
        ),
        (
            np.array([[0.5, -0.25], [1.5, 2.0]]),
            np.array([[0.01, -0.03], [0.2, -0.1]]),
            np.array([[0.1, 0.0], [-0.02, 0.04]]),
            np.array([[0.001, 0.002], [0.003, 0.004]]),
            12,
            3e-3,
            0.92,
            0.9995,
            1e-8,
            0.001,
        ),
    ]

    worst = 0.0
    for case in cases:
        expected = _adamw_oracle(*case)
        try:
            got = sol.offloaded_adamw_step(*case)
        except Exception:
            return {"max_abs_err": float("inf")}

        if len(got) != 3:
            return {"max_abs_err": float("inf")}

        for a, b in zip(got, expected):
            err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            worst = max(worst, err)

    return {"max_abs_err": worst}
