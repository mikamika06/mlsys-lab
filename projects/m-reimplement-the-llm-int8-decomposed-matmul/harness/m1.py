import ref
import numpy as np


def check(workdir):
    from int8_matmul.outliers import compute_outlier_curve
    t = ref.get_test_tensor()
    thresholds = ref.get_thresholds()
    want = ref.ref_compute_curve(t, thresholds)
    try:
        got = compute_outlier_curve(t, thresholds)
    except Exception as e:
        return {"curve_matched": 0.0, "_note": f"raised {type(e).__name__}: {e}"}

    if not isinstance(got, (list, np.ndarray)) or len(got) != len(want):
        return {"curve_matched": 0.0, "_note": f"got length {len(got) if hasattr(got, '__len__') else 'non-len'}, want {len(want)}"}

    match = 1.0 if np.allclose(got, want, atol=1e-5) else 0.0
    out = {"curve_matched": match}
    if match == 0.0:
        out["_note"] = f"got {got}, want {want}"
    return out
