import numpy as np
import ref


def check(workdir):
    from fp8.descale import compute_scale, quantize_and_descale
    from fp8.optimize import find_optimal_scale

    out = {"descale_matched": 0.0, "scale_search_matched": 0.0}

    tensors = ref.generate_test_tensors()

    descale_ok = True
    for t in tensors:
        scale = compute_scale(t)
        ref_scale = ref.compute_scale(t)
        if not np.isclose(scale, ref_scale):
            descale_ok = False
            out["_note"] = f"compute_scale failed: got {scale}, expected {ref_scale}"
            break

        ref_q, ref_rec = ref.quantize_and_descale(t, scale)
        got_q, got_rec = quantize_and_descale(t, scale)

        if not np.array_equal(ref_q, got_q) or not np.allclose(ref_rec, ref_rec, equal_nan=True):
            descale_ok = False
            out["_note"] = "quantize_and_descale outputs differ from reference"
            break

    if descale_ok:
        out["descale_matched"] = 1.0

    candidates = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    search_ok = True
    for t in tensors[:2]:
        ref_s, ref_mse = ref.find_optimal_scale(t, candidates)
        got_s, got_mse = find_optimal_scale(t, candidates)
        if not np.isclose(ref_s, got_s) or not np.isclose(ref_mse, got_mse):
            search_ok = False
            out["_note"] = f"find_optimal_scale mismatch: got ({got_s}, {got_mse}), expected ({ref_s}, {ref_mse})"
            break

    if search_ok:
        out["scale_search_matched"] = 1.0

    return out
