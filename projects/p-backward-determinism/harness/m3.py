import numpy as np

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from det.kernels import deterministic_backward
    m = {"deterministic_kernel": 0.0}
    try:
        g = np.array([1.123456, 2.654321])
        res1 = deterministic_backward(g)
        res2 = deterministic_backward(g)
        if np.allclose(res1, res2):
            m["deterministic_kernel"] = 1.0
    except Exception:
        pass
    return m
