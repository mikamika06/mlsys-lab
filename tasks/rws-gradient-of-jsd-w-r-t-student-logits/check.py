import numpy as np


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def _jsd_beta(p, q, beta):
    m = beta * p + (1.0 - beta) * q
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return beta * kl_pm + (1.0 - beta) * kl_qm


def _fd_grad(teacher_logits, student_logits, beta, eps=1e-5):
    p = _softmax(teacher_logits)
    V = student_logits.shape[0]
    grad = np.zeros(V, dtype=np.float64)
    for k in range(V):
        zp = student_logits.copy()
        zp[k] += eps
        zm = student_logits.copy()
        zm[k] -= eps
        fp = _jsd_beta(p, _softmax(zp), beta)
        fm = _jsd_beta(p, _softmax(zm), beta)
        grad[k] = (fp - fm) / (2.0 * eps)
    return grad


def _cases():
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(6):
        V = int(rng.integers(5, 11))
        t = rng.standard_normal(V) * 2.0
        s = rng.standard_normal(V) * 2.0
        beta = float(rng.uniform(0.1, 0.9))
        cases.append((t, s, beta))
    return cases


def grade(sol, fx) -> dict:
    worst = 0.0
    for t, s, beta in _cases():
        ref = _fd_grad(t, s, beta)
        try:
            got = np.asarray(
                sol.jsd_grad_wrt_student_logits(t.copy(), s.copy(), beta),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": worst}
