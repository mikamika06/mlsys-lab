import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _oracle_loss(teacher_logits, student_logits, teacher_hidden, student_hidden, beta):
    p = _softmax(teacher_logits)
    q = _softmax(student_logits)
    kl = np.sum(p * (np.log(p) - np.log(q)))
    mse = beta * np.mean((student_hidden - teacher_hidden) ** 2)
    return float(kl + mse)


def _finite_diff_logits(tl, sl, th, sh, beta, eps=1e-6):
    grad = np.zeros_like(sl, dtype=np.float64)
    for idx in np.ndindex(sl.shape):
        plus = sl.copy()
        minus = sl.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (
            _oracle_loss(tl, plus, th, sh, beta)
            - _oracle_loss(tl, minus, th, sh, beta)
        ) / (2 * eps)
    return grad


def _finite_diff_hidden(tl, sl, th, sh, beta, eps=1e-6):
    grad = np.zeros_like(sh, dtype=np.float64)
    for idx in np.ndindex(sh.shape):
        plus = sh.copy()
        minus = sh.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (
            _oracle_loss(tl, sl, th, plus, beta)
            - _oracle_loss(tl, sl, th, minus, beta)
        ) / (2 * eps)
    return grad


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[2.0, 0.5, -1.0], [0.1, 1.2, 0.3]]),
            np.array([[1.7, 0.8, -0.7], [0.2, 0.9, 0.5]]),
            np.array([[1.0, -1.0], [0.5, 2.0]]),
            np.array([[0.8, -0.5], [0.2, 1.7]]),
            0.2,
        ),
        (
            np.array([[0.4, -0.2]]),
            np.array([[0.1, 0.3]]),
            np.array([[2.0, 1.0, -1.0]]),
            np.array([[1.5, 0.7, -0.4]]),
            0.05,
        ),
    ]

    loss_err = 0.0
    grad_err = 0.0

    for tl, sl, th, sh, beta in cases:
        try:
            loss, got_gl, got_gh = sol.combined_logit_intermediate_loss(
                tl.copy(), sl.copy(), th.copy(), sh.copy(), beta
            )
        except Exception:
            return {"loss_max_abs_err": float("inf"), "grad_max_abs_err": float("inf")}

        ref_loss = _oracle_loss(tl, sl, th, sh, beta)
        ref_gl = _finite_diff_logits(tl, sl, th, sh, beta)
        ref_gh = _finite_diff_hidden(tl, sl, th, sh, beta)

        loss_err = max(loss_err, abs(float(loss) - ref_loss))
        grad_err = max(
            grad_err,
            float(np.max(np.abs(np.asarray(got_gl) - ref_gl))),
            float(np.max(np.abs(np.asarray(got_gh) - ref_gh))),
        )

    return {
        "loss_max_abs_err": loss_err,
        "grad_max_abs_err": grad_err,
    }
