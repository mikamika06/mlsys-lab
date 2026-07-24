import sys
import numpy as np


def _trace_counter(filename):
    count = {"n": 0}

    def tracer(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == filename:
            count["n"] += 1
        return tracer

    return tracer, count


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[1.0, 2.0, 3.0]]),
            np.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]]),
            np.array([[10.0, 10.0, 10.0]]),
        ),
        (
            np.arange(6, dtype=np.float64).reshape(2, 3),
            np.array([[2.0, 3.0, 4.0]]),
            np.array([[1.0], [5.0]]),
        ),
        (
            np.array([[5.0]]),
            np.arange(12, dtype=np.float64).reshape(4, 3),
            np.ones((4, 1)),
        ),
    ]

    max_err = 0.0
    line_events = 0

    for A, B, C in cases:
        ref = np.asarray(A) + np.asarray(B) * np.asarray(C)

        try:
            tracer, count = _trace_counter(sol.broadcast_add_mul.__code__.co_filename)
            sys.settrace(tracer)
            got = sol.broadcast_add_mul(A, B, C)
            sys.settrace(None)
            line_events += count["n"]
        except Exception:
            sys.settrace(None)
            return {"max_abs_err": float("inf"), "line_events": 0}

        got = np.asarray(got, dtype=np.float64)
        if got.shape != ref.shape:
            max_err = float("inf")
            continue
        err = float(np.max(np.abs(got - ref)))
        max_err = max(max_err, err)

    return {
        "max_abs_err": max_err,
        "line_events": line_events,
    }
