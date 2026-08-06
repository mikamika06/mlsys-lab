import ref
import numpy as np

def check(workdir):
    from metal_kernels.kernel import run_sum_reduction_kernel
    out = {"drift_analyzed": 0.0, "safe_mode_matches": 0.0}
    try:
        arr = np.linspace(0.1, 10.0, 2048, dtype=np.float32)
        want = ref.compute_reference_sum(arr)
        fast_res = run_sum_reduction_kernel(arr, math_mode="fast")
        safe_res = run_sum_reduction_kernel(arr, math_mode="safe")
        if safe_res is not None and np.isclose(safe_res, want, rtol=1e-4, atol=1e-4):
            out["safe_mode_matches"] = 1.0
        if fast_res is not None and safe_res is not None:
            out["drift_analyzed"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
