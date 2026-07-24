import numpy as np


def _oracle(teacher, student, projection):
    teacher = np.asarray(teacher, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    projected = student @ projection
    return float(np.mean((teacher - projected) ** 2))


def _rel_err(a, b):
    return abs(float(a) - float(b)) / (abs(float(b)) + 1e-12)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    cases = [
        (
            rng.normal(size=(4, 3)),
            rng.normal(size=(4, 2)),
            rng.normal(size=(2, 3)),
        ),
        (
            rng.normal(size=(16, 8)),
            rng.normal(size=(16, 5)),
            rng.normal(size=(5, 8)),
        ),
        (
            np.zeros((3, 2)),
            np.ones((3, 4)),
            np.zeros((4, 2)),
        ),
    ]

    worst = 0.0
    for teacher, student, projection in cases:
        try:
            got = sol.hidden_distillation_loss(
                teacher.copy(), student.copy(), projection.copy()
            )
        except Exception:
            return {"rel_err": 1.0}
        ref = _oracle(teacher, student, projection)
        worst = max(worst, _rel_err(got, ref))
    return {"rel_err": worst}
