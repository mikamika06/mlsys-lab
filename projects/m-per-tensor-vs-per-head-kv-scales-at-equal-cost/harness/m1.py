import ref
import numpy as np


def check(workdir):
    from kvquant.scales import compute_scales

    tensors = ref.get_test_tensors()
    matched = 0
    for t in tensors:
        for mode in ["per-tensor", "per-head"]:
            want = ref.compute_ref_scales(t, mode)
            got = compute_scales(t, mode)
            if np.allclose(want, got, atol=1e-5):
                matched += 1
    total = len(tensors) * 2
    ok = 1.0 if matched == total else 0.0
    return {"mse_matched": ok}
