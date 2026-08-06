import ref
import numpy as np

def check(workdir):
    from metal_kernels.kernel import run_indexing_kernel
    out = {"indexing_correct": 0.0}
    try:
        size = 1024
        res = run_indexing_kernel(size)
        want = ref.compute_reference_indexing(size)
        if res is not None and np.allclose(res, want, atol=1e-5):
            out["indexing_correct"] = 1.0
        else:
            out["_note"] = f"got {res[:5] if res is not None else None}, want {want[:5]}"
    except Exception as e:
        out["_note"] = str(e)
    return out
