import ref
import numpy as np


def check(workdir):
    from quanteval.eval import compute_relative_error
    out = {"rel_err_match": 0.0}
    ok = 0
    for case in ref.CASES:
        want = ref.compute_relative_error(case["fp32_out"], case["int8_out"])
        got = compute_relative_error(case["fp32_out"], case["int8_out"])
        if np.isclose(got, want, rtol=1e-5, atol=1e-5):
            ok += 1
        else:
            out["_note"] = f"got rel_err {got}, want {want}"
    if ok == len(ref.CASES):
        out["rel_err_match"] = 1.0
    return out
