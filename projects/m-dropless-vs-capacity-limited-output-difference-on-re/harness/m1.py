import ref
import numpy as np

def check(workdir):
    from moe_routing.analysis import compare_outputs
    tokens, logits, top_k = ref.generate_test_data(seed=123)
    cf = 1.0
    want = ref.compute_capacity_limited_output(tokens, logits, top_k, cf)
    got = compare_outputs(tokens, logits, top_k, cf)

    err = float(np.max(np.abs(want - got)))
    return {"max_abs_err": err}
