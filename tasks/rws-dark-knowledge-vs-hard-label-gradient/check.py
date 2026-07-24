import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _oracle(student_logits, teacher_probs, label, temperature):
    p_kd = _softmax(student_logits / temperature)
    kd_grad = temperature * (p_kd - np.asarray(teacher_probs, dtype=np.float64))

    p_ce = _softmax(student_logits)
    y = np.zeros_like(p_ce)
    y[int(label)] = 1.0
    ce_grad = p_ce - y
    return kd_grad, ce_grad


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([2.0, 1.0, 0.0]),
            np.array([0.70, 0.20, 0.10]),
            0,
            2.0,
        ),
        (
            np.array([0.5, 1.5, -0.25, 0.0]),
            np.array([0.10, 0.60, 0.20, 0.10]),
            1,
            3.0,
        ),
    ]

    max_err = 0.0
    direction_diff = 0.0

    try:
        for student, teacher, label, temperature in cases:
            ref_kd, ref_ce = _oracle(student, teacher, label, temperature)
            got_kd, got_ce = sol.kd_hard_label_grad(
                student.copy(),
                teacher.copy(),
                label,
                temperature,
            )
            err = max(
                float(np.max(np.abs(np.asarray(got_kd, dtype=np.float64) - ref_kd))),
                float(np.max(np.abs(np.asarray(got_ce, dtype=np.float64) - ref_ce))),
            )
            max_err = max(max_err, err)

            a = np.asarray(got_kd, dtype=np.float64)
            b = np.asarray(got_ce, dtype=np.float64)
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            direction_diff = max(direction_diff, cosine)
    except Exception:
        return {"max_abs_err": 1.0, "direction_diff": 1.0}

    return {
        "max_abs_err": max_err,
        "direction_diff": direction_diff,
    }
