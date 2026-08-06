import ref
import numpy as np

def check(workdir):
    from speculative.analysis import compare_heuristics
    out = {"tv_compared": 0.0}
    p = ref.TEST_CASES[0]["p"]
    q = ref.TEST_CASES[0]["q"]
    res = compare_heuristics(p, q)
    ref_res = ref.compare_heuristics(p, q)
    if abs(res["tv_speculative"] - ref_res["tv_speculative"]) < 1e-5 and \
       abs(res["tv_top1"] - ref_res["tv_top1"]) < 1e-5:
        out["tv_compared"] = 1.0
    else:
        out["_note"] = f"got {res}, expected {ref_res}"
    return out
