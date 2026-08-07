import ref
import numpy as np

def check(workdir):
    m = {"isolated_test_ok": 0.0}
    try:
        res = ref.oracle_isolate()
        if np.allclose(res, [20.0]):
            m["isolated_test_ok"] = 1.0
    except Exception:
        pass
    return m
