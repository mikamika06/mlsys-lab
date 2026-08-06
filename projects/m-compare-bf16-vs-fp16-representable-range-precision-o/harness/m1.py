import ref
import numpy as np

def check(workdir):
    from dtypecheck.analysis import analyze_ranges
    tensors = ref.generate_extreme_tensors()
    max_err = 0.0
    for t in tensors:
        ref_res = ref.analyze_ranges(t)
        got_res = analyze_ranges(t)
        err = np.max(np.abs(ref_res["bf16_approx"] - got_res["bf16_approx"]))
        if err > max_err:
            max_err = float(err)
    return {"rel_err": max_err}
