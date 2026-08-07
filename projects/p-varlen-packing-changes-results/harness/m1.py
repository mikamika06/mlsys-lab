import numpy as np
import ref

def check(workdir):
    from pack.attention import detect_boundary
    m = {"boundary_detected": 0.0}
    q = np.random.randn(16, 8)
    cu_seqlens = np.array([0, 5, 12, 16], dtype=np.int32)
    res = detect_boundary(q, cu_seqlens)
    if res is not None and len(res) == len(cu_seqlens) - 1:
        m["boundary_detected"] = 1.0
    return m
