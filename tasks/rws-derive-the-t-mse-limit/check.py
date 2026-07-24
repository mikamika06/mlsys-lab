import numpy as np


def _oracle(z_teacher, z_student, temperatures):
    z_teacher = np.asarray(z_teacher, dtype=np.float64)
    z_student = np.asarray(z_student, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)

    def stable_softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    scaled = []
    for t in temperatures:
        pt = stable_softmax(z_teacher / t)
        ps = stable_softmax(z_student / t)
        kl = np.sum(pt * (np.log(pt) - np.log(ps)))
        scaled.append((t * t) * kl)

    zt = z_teacher - np.mean(z_teacher)
    zs = z_student - np.mean(z_student)
    limit = 0.5 * np.sum((zt - zs) ** 2)

    return np.asarray(scaled, dtype=np.float64), float(limit)


def _rel_err(a, b):
    return float(
        np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        / (np.linalg.norm(np.asarray(b, dtype=np.float64)) + 1e-12)
    )


def grade(sol, fx) -> dict:
    z_teacher = np.array([3.2, -0.4, 1.7, 0.1], dtype=np.float64)
    z_student = np.array([2.9, -0.1, 1.2, 0.4], dtype=np.float64)
    temperatures = np.array([1.0, 2.0, 10.0, 100.0, 1000.0], dtype=np.float64)

    ref_scaled, ref_limit = _oracle(z_teacher, z_student, temperatures)

    try:
        got_scaled, got_limit = sol.t_mse_limit(
            z_teacher.copy(),
            z_student.copy(),
            temperatures.copy(),
        )
    except Exception:
        return {"scaled_kl_rel_err": 1.0, "limit_rel_err": 1.0}

    return {
        "scaled_kl_rel_err": _rel_err(got_scaled, ref_scaled),
        "limit_rel_err": _rel_err(got_limit, ref_limit),
    }
