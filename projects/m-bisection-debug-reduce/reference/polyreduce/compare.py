import numpy as np
from polyreduce.sanitize import sanitize_tensor

def classify_divergence(arr_a, arr_b, rtol=1e-3, atol=1e-5, denormal_threshold=1e-7):
    """Classify comparison as MATCH, FP_NOISE, or REAL_BUG."""
    a = np.asarray(arr_a)
    b = np.asarray(arr_b)
    if a.shape != b.shape:
        return "REAL_BUG"
    raw_match = bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    if raw_match:
        return "MATCH"
    san_a = sanitize_tensor(a, denormal_threshold=denormal_threshold)
    san_b = sanitize_tensor(b, denormal_threshold=denormal_threshold)
    san_match = bool(np.allclose(san_a, san_b, rtol=rtol, atol=atol, equal_nan=True))
    if san_match:
        return "FP_NOISE"
    return "REAL_BUG"
