import numpy as np
import ref

def check(workdir):
    from pack.attention import varlen_attention
    m = {"tolerance_ok": 0.0}
    np.random.seed(100)
    q = np.random.randn(20, 16)
    k = np.random.randn(20, 16)
    v = np.random.randn(20, 16)
    cu_seqlens = np.array([0, 7, 15, 20], dtype=np.int32)
    out = varlen_attention(q, k, v, cu_seqlens)
    expected = ref.compute_reference(q, k, v, cu_seqlens)
    diff = np.max(np.abs(out - expected))
    if diff < 1e-4:
        m["tolerance_ok"] = 1.0
    return m
