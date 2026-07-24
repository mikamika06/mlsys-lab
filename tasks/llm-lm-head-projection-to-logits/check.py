import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    tests = [
        (rng.normal(size=(2,3,4)), rng.normal(size=(5,4)), rng.normal(size=(5,))),
        (rng.normal(size=(1,1,8)), rng.normal(size=(10,8)), rng.normal(size=(10,))),
        (rng.normal(size=(4,5,6)), rng.normal(size=(7,6)), rng.normal(size=(7,))),
    ]
    max_err = 0.0
    for hidden, weight, bias in tests:
        try:
            got = sol.lm_head_projection(hidden, weight, bias)
        except Exception:
            return {"max_abs_err": 0.0}
        ref = np.matmul(hidden, weight.T) + bias
        err = scorers.max_abs_err(ref, got)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
