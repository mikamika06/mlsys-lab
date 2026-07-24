import sys
import numpy as np


def _oracle(A):
    A = np.asarray(A, dtype=np.float64)
    g = np.sum(A * A, axis=1)
    return g[:, None] + g[None, :] - 2.0 * (A @ A.T)


def _run_with_trace(fn, A):
    count = 0
    code = getattr(fn, "__code__", None)

    def tracer(frame, event, arg):
        nonlocal count
        if event == "line" and code is not None and frame.f_code is code:
            count += 1
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        out = fn(A)
    finally:
        sys.settrace(old)
    return out, count


def grade(sol, fx) -> dict:
    cases = [
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]),
        np.arange(40, dtype=np.float64).reshape(10, 4) / 7.0,
        np.sin(np.arange(99, dtype=np.float64).reshape(11, 9)),
    ]

    max_mse = 0.0
    max_lines = 0
    ok = 1.0

    for A in cases:
        ref = _oracle(A)
        try:
            got, lines = _run_with_trace(sol.pairwise_sq_dists, A)
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            ok = 0.0
            break

        if got.shape != ref.shape:
            ok = 0.0
            break

        mse = float(np.mean((got - ref) ** 2))
        max_mse = max(max_mse, mse)
        max_lines = max(max_lines, lines)

        if mse > 1e-8:
            ok = 0.0
        if lines > 60:
            ok = 0.0

    return {
        "mse": max_mse if ok else max(max_mse, 1.0),
        "op_count": float(max_lines)
    }
