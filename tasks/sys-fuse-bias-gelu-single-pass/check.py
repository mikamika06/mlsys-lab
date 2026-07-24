import numpy as np
import sys

def _gelu(x):
    sqrt2pi = np.sqrt(2/np.pi)
    return sqrt2pi * x * (1 + np.tanh(sqrt2pi*(x + 0.044715*x**3)))

def _ref(x, bias):
    y = x + bias
    return _gelu(y)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    n = 1000
    x = rng.standard_normal(n).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    try:
        got = sol.fuse_bias_gelu(x, bias)
    except Exception:
        return {"max_abs_err": 1.0, "op_count": float("inf")}
    ref = _ref(x, bias)
    max_abs_err = np.max(np.abs(got - ref))
    counter = 0
    def tracer(frame, event, arg):
        nonlocal counter
        if event == "line" and frame.f_code.co_name == "fuse_bias_gelu":
            counter += 1
        return tracer
    sys.settrace(tracer)
    try:
        sol.fuse_bias_gelu(x, bias)
    finally:
        sys.settrace(None)
    op_count = counter
    return {"max_abs_err": float(max_abs_err), "op_count": op_count}
