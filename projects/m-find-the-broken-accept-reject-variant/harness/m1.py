import ref
import numpy as np

def check(workdir):
    from speculative.sampling import find_correct_variant, argmin_index
    out = {"variant_identified": 0.0, "argmin_index": 0.0}
    ok = 0
    for case in ref.TEST_CASES:
        got = find_correct_variant(case["variants"], case["p"], case["q"])
        correct_idx = ref.find_correct_variant(case["variants"], case["p"], case["q"])
        if got == correct_idx:
            ok += 1
    out["variant_identified"] = 1.0 if ok == len(ref.TEST_CASES) else 0.0
    dummy_vals = np.array([5.0, 1.2, 3.4])
    if argmin_index(dummy_vals) == 1:
        out["argmin_index"] = 1.0
    return out
