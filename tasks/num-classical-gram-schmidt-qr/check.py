import sys
import numpy as np


def _called_forbidden_qr(fn):
    called = {"bad": False}

    def tracer(frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            mod = frame.f_globals.get("__name__", "")
            if name == "qr" and mod.startswith("numpy.linalg"):
                called["bad"] = True
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        fn()
    finally:
        sys.settrace(old)
    return called["bad"]


def _reference(A):
    q, r = np.linalg.qr(A, mode="reduced")
    return q, r


def _align_signs(q, r, q_ref, r_ref):
    q = q.copy()
    r = r.copy()
    for i in range(r.shape[0]):
        if r_ref[i, i] * r[i, i] < 0:
            q[:, i] *= -1
            r[i, :] *= -1
    return q, r


def grade(sol, fx) -> dict:
    cases = [
        np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([
            [2.0, -1.0, 3.0],
            [0.0, 4.0, 1.0],
            [5.0, 2.0, -2.0],
            [1.0, 0.5, 2.0],
        ]),
        np.array([
            [3.0, 2.0],
            [1.0, 4.0],
            [2.0, -3.0],
        ]),
    ]

    worst = 0.0
    for A in cases:
        try:
            if _called_forbidden_qr(lambda: sol.classical_gram_schmidt_qr(A.copy())):
                return {"max_abs_err": float("inf")}

            q, r = sol.classical_gram_schmidt_qr(A.copy())
            if not isinstance(q, np.ndarray) or not isinstance(r, np.ndarray):
                return {"max_abs_err": float("inf")}

            q_ref, r_ref = _reference(A)
            q, r = _align_signs(q, r, q_ref, r_ref)

            err = max(
                float(np.max(np.abs(q - q_ref))),
                float(np.max(np.abs(r - r_ref))),
            )
            worst = max(worst, err)
        except Exception:
            return {"max_abs_err": float("inf")}

    return {"max_abs_err": worst}
