import numpy as np
import ref


def check(workdir):
    from fused_attn.kernel import tiled_attention
    m = {"causal_ok": 0.0}
    np.random.seed(456)
    q = np.random.randn(64, 32)
    k = np.random.randn(64, 32)
    v = np.random.randn(64, 32)
    try:
        ref_out = ref.compute_reference_attention(q, k, v, causal=True)
        out = tiled_attention(q, k, v, block_size=16, causal=True)
        if np.allclose(ref_out, out, atol=1e-5, rtol=1e-5):
            m["causal_ok"] = 1.0
    except Exception:
        pass
    return m
