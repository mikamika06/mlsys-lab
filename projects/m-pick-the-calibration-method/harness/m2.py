import numpy as np
import ref

def check(workdir):
    from calib.collapse import detect_scale_collapse

    out = {"collapse_detected": 0.0, "scale_ratio_matched": 0.0}

    np.random.seed(42)
    counts, bin_edges = np.histogram(np.random.normal(0, 1, 1000), bins=100, range=(-100, 100))
    histogram = (counts, bin_edges)

    minmax_range = (-100.0, 100.0)
    target_range = (-3.0, 3.0)

    want = ref.detect_scale_collapse(histogram, minmax_range, target_range)
    got = detect_scale_collapse(histogram, minmax_range, target_range)

    if got.get("collapsed") == want["collapsed"]:
        out["collapse_detected"] = 1.0
    else:
        out["_note"] = f"collapsed mismatch: got {got.get('collapsed')}, want {want['collapsed']}"

    if abs(got.get("scale_ratio", 0.0) - want["scale_ratio"]) < 1e-5:
        out["scale_ratio_matched"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"scale_ratio mismatch: got {got.get('scale_ratio')}, want {want['scale_ratio']}"

    return out
