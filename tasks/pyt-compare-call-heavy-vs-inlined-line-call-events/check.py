import sys

import numpy as np

from mlsys.scorers import rel_err


def _count_call_events(fn, *args, **kwargs) -> int:
    n = 0

    def tracer(frame, event, arg):
        nonlocal n
        if event == "call":
            n += 1
        return tracer

    prev = sys.gettrace()
    sys.settrace(tracer)
    try:
        fn(*args, **kwargs)
    finally:
        sys.settrace(prev)
    return n


def _fixture():
    rng = np.random.default_rng(0)
    return rng.standard_normal(200)


def grade(sol, fx) -> dict:
    x = _fixture()
    ref = float(np.sum(x ** 2))

    try:
        heavy_out = float(sol.sum_squares_call_heavy(x.copy()))
        inline_out = float(sol.sum_squares_inlined(x.copy()))
    except Exception:
        return {"rel_err": float("inf"), "call_heavy_events": 0.0, "call_inline_events": 999.0}

    if not (np.isfinite(heavy_out) and np.isfinite(inline_out)):
        return {"rel_err": float("inf"), "call_heavy_events": 0.0, "call_inline_events": 999.0}

    err = rel_err(np.array([ref, ref]), np.array([heavy_out, inline_out]))

    # warm up (import/first-call overhead) before the probed calls
    try:
        sol.sum_squares_call_heavy(x.copy())
        sol.sum_squares_inlined(x.copy())
        heavy_calls = _count_call_events(sol.sum_squares_call_heavy, x.copy())
        inline_calls = _count_call_events(sol.sum_squares_inlined, x.copy())
    except Exception:
        return {"rel_err": float("inf"), "call_heavy_events": 0.0, "call_inline_events": 999.0}

    return {
        "rel_err": float(err),
        "call_heavy_events": float(heavy_calls),
        "call_inline_events": float(inline_calls),
    }
