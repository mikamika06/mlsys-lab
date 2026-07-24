import sys

import numpy as np


def _calls_forbidden_svd(fn):
    called = {"bad": False}

    def tracer(frame, event, arg):
        if event == "call":
            name = frame.f_code.co_name
            mod = frame.f_globals.get("__name__", "")
            if name == "svd" and mod.startswith("numpy.linalg"):
                called["bad"] = True
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = fn()
    finally:
        sys.settrace(old)
    return called["bad"], result


def _oracle(A: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.svd(A, compute_uv=False).astype(np.float64))[::-1]


def _rel_err(got: np.ndarray, ref: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        rng.standard_normal((4, 4)),
        rng.standard_normal((6, 3)),   # tall
        rng.standard_normal((3, 6)),   # wide
        rng.standard_normal((5, 5)),
        rng.standard_normal((2, 7)),
        np.array([[3.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
    ]

    worst = 0.0
    for A in cases:
        ref = _oracle(A)
        try:
            bad, got = _calls_forbidden_svd(lambda: sol.svd_singular_values(A.copy()))
            if bad:
                return {"rel_err": float("inf")}
            got = np.sort(np.asarray(got, dtype=np.float64))[::-1]
            if got.shape != ref.shape:
                return {"rel_err": float("inf")}
        except Exception:
            return {"rel_err": float("inf")}
        worst = max(worst, _rel_err(got, ref))

    return {"rel_err": worst}
