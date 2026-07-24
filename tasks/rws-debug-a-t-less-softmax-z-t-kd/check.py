import numpy as np
from mlsys import scorers


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def _oracle(teacher_logits, student_logits, T):
    teacher_logits = np.asarray(teacher_logits, dtype=np.float64)
    student_logits = np.asarray(student_logits, dtype=np.float64)
    n = teacher_logits.shape[0]

    p = _softmax(teacher_logits / T)
    q = _softmax(student_logits / T)

    loss = T * T * (-np.sum(p * np.log(q)) / n)
    grad = T * (q - p) / n
    return loss, grad


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[3.0, 1.0, -2.0], [0.5, -1.0, 2.0]]),
            np.array([[2.0, 0.0, -1.0], [-0.5, 1.5, 0.5]]),
            2.0,
        ),
        (
            np.array([[1.2, -0.7, 0.4, 2.1]]),
            np.array([[0.2, 1.1, -1.5, 0.3]]),
            5.0,
        ),
        (
            np.array([[0.1, 0.2], [2.0, -3.0], [-1.0, 1.0]]),
            np.array([[0.0, -0.1], [1.5, -2.5], [0.5, 0.2]]),
            1.5,
        ),
    ]

    errors = []
    for teacher, student, T in cases:
        try:
            got_loss, got_grad = sol.kd_loss_and_grad(
                teacher.copy(), student.copy(), T
            )
        except Exception:
            return {"rel_err": 1.0}

        ref_loss, ref_grad = _oracle(teacher, student, T)
        got = np.concatenate(
            [
                np.asarray([got_loss], dtype=np.float64),
                np.asarray(got_grad, dtype=np.float64).ravel(),
            ]
        )
        ref = np.concatenate(
            [
                np.asarray([ref_loss], dtype=np.float64),
                np.asarray(ref_grad, dtype=np.float64).ravel(),
            ]
        )
        errors.append(scorers.rel_err(ref, got))

    return {"rel_err": float(max(errors))}
