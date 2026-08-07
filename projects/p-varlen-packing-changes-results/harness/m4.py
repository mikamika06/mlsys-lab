import numpy as np
import ref

def check(workdir):
    from pack.attention import varlen_attention
    m = {"equivalence_ok": 0.0}
    np.random.seed(42)
    q = np.random.randn(12, 8)
    k = np.random.randn(12, 8)
    v = np.random.randn(12, 8)
    cu_seqlens = np.array([0, 4, 8, 12], dtype=np.int32)
    out = varlen_attention(q, k, v, cu_seqlens)
    expected = ref.compute_reference(q, k, v, cu_seqlens)
    if np.allclose(out, expected, atol=1e-5):
        m["equivalence_ok"] = 1.0
    return m
