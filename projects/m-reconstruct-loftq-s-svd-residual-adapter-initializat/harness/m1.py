import ref
import numpy as np


def check(workdir):
    from loftq.residual import init_loftq_residual

    out = {"residual_matched": 0.0}
    ok = 0
    for case in ref.CASES:
        W = case["W"]
        r = case["rank"]
        want_wq, want_A, want_B = ref.loftq_init(W, r)
        got_wq, got_A, got_B = init_loftq_residual(W, r)
        if np.allclose(want_wq, got_wq, atol=1e-5) and np.allclose(want_A, got_A, atol=1e-5) and np.allclose(want_B, got_B, atol=1e-5):
            ok += 1
    if ok == len(ref.CASES):
        out["residual_matched"] = 1.0
    else:
        out["_note"] = f"Matched {ok}/{len(ref.CASES)} cases for LoftQ residual initialization"
    return out
