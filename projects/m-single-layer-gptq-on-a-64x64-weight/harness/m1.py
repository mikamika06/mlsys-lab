import ref
import numpy as np

def check(workdir):
    from gptq_single.gptq import compute_hessian_inverse
    W, X = ref.get_data()
    want = ref.compute_hessian_inv(X)
    got = compute_hessian_inverse(X)
    match = 1.0 if np.allclose(want, got, atol=1e-5) else 0.0
    out = {"hessian_match": match}
    if match == 0.0:
        out["_note"] = f"hessian inverse mismatch max diff: {np.max(np.abs(want - got))}"
    return out
