import sys
import numpy as np

def check(workdir):
    sys.path.insert(0, workdir)
    from exporter.replacements import check_tolerance
    m = {"tolerance_ok": 0.0}
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0001, 2.0001, 3.0001], dtype=np.float32)
    if check_tolerance(a, b, 1e-3):
        m["tolerance_ok"] = 1.0
    return m
