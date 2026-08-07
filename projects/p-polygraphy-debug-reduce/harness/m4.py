import ref
import numpy as np

def check(workdir):
    m = {"patch_ok": 0.0}
    try:
        res = ref.oracle_patch()
        if np.allclose(res, [3.0]):
            m["patch_ok"] = 1.0
    except Exception:
        pass
    return m
