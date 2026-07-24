import numpy as np
import sys

def _swish(z):
    return z / (1 + np.exp(-z))

def grade(sol, fx) -> dict:
    # deterministic random test case
    rng = np.random.default_rng(0)
    n, d, h = 5, 4, 3
    x     = rng.standard_normal((n, d))
    w_up  = rng.standard_normal((d, h))
    b_up  = rng.standard_normal(h)
    w_gate= rng.standard_normal((d, h))
    b_gate= rng.standard_normal(h)

    # reference output using NumPy directly
    ref_u   = x @ w_up + b_up
    ref_g   = _swish(x @ w_gate + b_gate)
    ref_out = ref_u * ref_g

    # trace line events inside the student's function
    counter = [0]
    def tracer(frame, event, arg):
        if event == "line":
            code = frame.f_code
            if (code.co_name == sol.fused_swiglu.__name__ and
                code.co_filename == sol.__file__):
                counter[0] += 1
        return tracer

    sys.settrace(tracer)
    try:
        out = sol.fused_swiglu(x, w_up, b_up, w_gate, b_gate)
    finally:
        sys.settrace(None)

    # compute metrics
    max_abs_err = np.max(np.abs(out - ref_out))
    op_count    = counter[0]
    return {"max_abs_err": float(max_abs_err), "op_count": int(op_count)}
