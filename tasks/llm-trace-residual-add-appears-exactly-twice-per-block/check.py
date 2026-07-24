import sys
import numpy as np
from mlsys import scorers

def _reference(x, w1, b1, w2, b2):
    a = x @ w1 + b1
    y = x + a
    b = y @ w2 + b2
    z = y + b
    return z.astype("float64")

def grade(sol, fx) -> dict:
    # generate deterministic test data
    rng = np.random.default_rng(0)
    n, d = 4, 5
    x = rng.standard_normal((n, d))
    w1 = rng.standard_normal((d, d))
    b1 = rng.standard_normal(d)
    w2 = rng.standard_normal((d, d))
    b2 = rng.standard_normal(d)

    # reference output
    ref = _reference(x, w1, b1, w2, b2)

    # trace counter for add_residual calls
    count = 0
    def tracer(frame, event, arg):
        nonlocal count
        if event == "call" and frame.f_code.co_name == "add_residual":
            count += 1
        return tracer

    sys.settrace(tracer)
    try:
        out = sol.transformer_block(x, w1, b1, w2, b2)
    finally:
        sys.settrace(None)

    err = scorers.rel_err(ref, out)
    return {"residual_add_count": count, "rel_err": err}
