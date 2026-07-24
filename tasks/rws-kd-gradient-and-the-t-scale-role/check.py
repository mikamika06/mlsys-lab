import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _loss(student, teacher, T, scale_t2):
    pt = _softmax(teacher / T)
    ps = _softmax(student / T)
    value = np.sum(pt * (np.log(pt + 1e-12) - np.log(ps + 1e-12)))
    if scale_t2:
        value *= T * T
    return float(value)


def _fd_grad(student, teacher, T, scale_t2):
    eps = 1e-6
    out = np.zeros_like(student, dtype=np.float64)
    for i in range(student.shape[0]):
        for j in range(student.shape[1]):
            a = student.copy()
            b = student.copy()
            a[i, j] += eps
            b[i, j] -= eps
            out[i, j] = (_loss(a, teacher, T, scale_t2) -
                         _loss(b, teacher, T, scale_t2)) / (2 * eps)
    return out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (rng.normal(size=(3, 5)), rng.normal(size=(3, 5)), np.array([0, 1, 2]), 2.0),
        (rng.normal(size=(4, 7)), rng.normal(size=(4, 7)), np.array([3, 2, 1, 0]), 5.0),
    ]

    max_err = 0.0
    for student, teacher, labels, T in cases:
        try:
            got = sol.kd_gradient(student, teacher, labels, T, True)
        except Exception:
            return {"max_abs_err": 1.0, "t2_scale_ratio": 0.0}
        ref = _fd_grad(student, teacher, T, True)
        max_err = max(max_err, float(np.max(np.abs(got - ref))))

    student = rng.normal(size=(5, 6))
    teacher = rng.normal(size=(5, 6))
    labels = np.arange(5)

    try:
        g2 = sol.kd_gradient(student, teacher, labels, 2.0, True)
        g8 = sol.kd_gradient(student, teacher, labels, 8.0, True)
        n2 = np.linalg.norm(g2)
        n8 = np.linalg.norm(g8)
        ratio_t2 = float(min(n2, n8) / max(n2, n8))

        g2_bad = sol.kd_gradient(student, teacher, labels, 2.0, False)
        g8_bad = sol.kd_gradient(student, teacher, labels, 8.0, False)
        shrink = float(np.linalg.norm(g8_bad) / (np.linalg.norm(g2_bad) + 1e-12))
    except Exception:
        ratio_t2 = 0.0
        shrink = 1.0

    scale_score = ratio_t2 if shrink < 0.3 else 0.0
    return {
        "max_abs_err": max_err,
        "t2_scale_ratio": scale_score,
    }
