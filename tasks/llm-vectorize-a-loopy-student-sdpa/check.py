import sys
import numpy as np


def _ref_sdpa(Q, K, V):
    d = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs @ V


def _trace_calls(fn, args):
    events = {"count": 0}
    code = fn.__code__

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code is code:
            events["count"] += 1
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        fn(*args)
    finally:
        sys.settrace(old)
    return events["count"]


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(123)
    cases = [
        (
            rng.normal(size=(4, 3)).astype(np.float64),
            rng.normal(size=(4, 3)).astype(np.float64),
            rng.normal(size=(4, 2)).astype(np.float64),
        ),
        (
            rng.normal(size=(8, 5)).astype(np.float64),
            rng.normal(size=(8, 5)).astype(np.float64),
            rng.normal(size=(8, 4)).astype(np.float64),
        ),
    ]

    max_err = 0.0
    for Q, K, V in cases:
        try:
            got = np.asarray(sol.sdpa(Q, K, V), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf"), "line_events": float("inf")}

        ref = _ref_sdpa(Q, K, V)
        max_err = max(max_err, float(np.max(np.abs(got - ref))))

    line_events = _trace_calls(sol.sdpa, cases[1])
    return {
        "max_abs_err": max_err,
        "line_events": float(line_events),
    }
