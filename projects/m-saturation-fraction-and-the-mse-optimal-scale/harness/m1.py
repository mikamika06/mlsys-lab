import ref
import numpy as np


def check(workdir):
    """Check milestone 1."""
    from quantlib.saturation import compute_mse_optimal_scale
    tensors, _, _ = ref.get_test_cases()
    match_count = 0
    total = len(tensors)
    for x in tensors:
        ref_scale = ref.compute_mse_optimal_scale(x)
        try:
            got_scale = compute_mse_optimal_scale(x)
            if np.isclose(got_scale, ref_scale, rtol=1e-2, atol=1e-2):
                match_count += 1
        except Exception:
            pass
    out = {"mse_scale_matched": 1.0 if match_count == total else 0.0}
    return out
