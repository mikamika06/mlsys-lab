import numpy as np
import ref

def check(workdir):
    out = {"sparsity_match": 0.0}
    try:
        from flexmask.sparsity import analyze_sparsity
    except Exception as e:
        out["_note"] = f"import error: {e}"
        return out

    mat = np.ones((32, 32), dtype=np.float32)
    mat[:16, :16] = 0
    res = analyze_sparsity(mat, block_size=16)
    want = ref.compute_sparsity_oracle(mat, block_size=16)

    if isinstance(res, dict) and "element_sparsity" in res and "block_sparsity" in res:
        if abs(res["element_sparsity"] - want["element_sparsity"]) < 1e-5 and \
           abs(res["block_sparsity"] - want["block_sparsity"]) < 1e-5:
            out["sparsity_match"] = 1.0
        else:
            out["_note"] = f"got sparsity {res}, expected {want}"
    else:
        out["_note"] = f"invalid return format: {res}"
    return out
