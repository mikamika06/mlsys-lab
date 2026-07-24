import numpy as np


def _oracle_yarn(q, k, inv_freq, positions, beta_fast, beta_slow, scale, temperature):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    inv_freq = np.asarray(inv_freq, dtype=np.float64)

    idx = np.arange(inv_freq.shape[0], dtype=np.float64)
    ramp = np.clip(
        (idx - beta_slow) / (beta_fast - beta_slow),
        0.0,
        1.0,
    )
    freq = inv_freq / (1.0 + ramp * (scale - 1.0))

    def rotate(x):
        out = np.empty_like(x, dtype=np.float64)
        angles = np.asarray(positions, dtype=np.float64)[:, None] * freq[None, :]
        c = np.cos(angles)
        s = np.sin(angles)
        a = x[:, 0::2]
        b = x[:, 1::2]
        out[:, 0::2] = a * c - b * s
        out[:, 1::2] = a * s + b * c
        return out

    qr = rotate(q)
    kr = rotate(k)
    return (qr @ kr.T) / (np.sqrt(q.shape[1]) * np.sqrt(temperature))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 0.0, 0.5, -0.5], [0.2, 0.8, -1.0, 0.3]]),
            np.array([[0.4, -0.2, 0.7, 0.1], [1.0, 0.3, 0.0, -0.4]]),
            np.array([1.0, 0.5]),
            np.array([1, 7]),
            1.5,
            0.25,
            4.0,
            0.7,
        ),
        (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
            np.array([[0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]),
            np.array([0.9, 0.4, 0.2]),
            np.array([13]),
            2.0,
            0.0,
            8.0,
            1.8,
        ),
    ]

    worst = 0.0
    for args in cases:
        ref = _oracle_yarn(*args)
        try:
            got = sol.yarn_ramp_temperature(*args)
            err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
