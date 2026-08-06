import sys
import numpy as np

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from polyreduce.sanitize import sanitize_tensor
        from polyreduce.compare import classify_divergence
    except Exception as e:
        return {"sanitize_correct": 0.0, "classify_correct": 0.0, "_note": f"Import error: {e}"}

    import ref

    san_cases = ref.generate_sanitize_cases()
    san_ok = True
    for arr, thresh, zero_nans, want in san_cases:
        got = sanitize_tensor(arr, denormal_threshold=thresh, zero_nans=zero_nans)
        if not np.allclose(got, want, equal_nan=True):
            san_ok = False

    cls_cases = ref.generate_classify_cases()
    cls_ok = True
    for a, b, rtol, atol, thresh, want in cls_cases:
        got = classify_divergence(a, b, rtol=rtol, atol=atol, denormal_threshold=thresh)
        if got != want:
            cls_ok = False

    return {
        "sanitize_correct": 1.0 if san_ok else 0.0,
        "classify_correct": 1.0 if cls_ok else 0.0
    }
