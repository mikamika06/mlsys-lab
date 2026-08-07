import ref
import numpy as np

def check(workdir):
    from gptq_single.gptq import gptq_quantize
    W, X = ref.get_data()
    want = ref.quantize_weights(W, ref.compute_hessian_inv(X))
    got = gptq_quantize(W, X)
    err = float(np.max(np.abs(want - got)))
    out = {"max_abs_err": err}
    if err > 1e-5:
        out["_note"] = f"max abs error too high: {err}"
    return out
