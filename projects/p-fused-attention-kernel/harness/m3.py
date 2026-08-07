import numpy as np
import ref


def check(workdir):
    from fused_attn.kernel import tiled_attention
    m = {"reference_match": 0.0}
    np.random.seed(123)
    q = np.random.randn(64, 32)
    k = np.random.randn(64, 32)
    v = np.random.randn(64, 32)
    try:
        ref_out = ref.compute_reference_attention(q, k, v, causal=False)
        out = tiled_attention(q, k, v, block_size=16, causal=False)
        if np.allclose(ref_out, out, atol=1e-5, rtol=1e-5):
            m["reference_match"] = 1.0
    except Exception:
        pass
    return m
