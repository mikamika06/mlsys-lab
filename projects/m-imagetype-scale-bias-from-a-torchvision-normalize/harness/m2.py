import ref
import numpy as np

def check(workdir):
    from imageconv.convert import verify_drift
    out = {"drift_verified": 0.0}
    arr1 = np.ones((4, 4), dtype=np.float32)
    arr2 = arr1 + 1e-7
    try:
        res = verify_drift(arr1, arr2, 1e-5)
        if res is True:
            out["drift_verified"] = 1.0
    except Exception:
        pass
    return out
