import numpy as np
import ref


def check(workdir):
    from fused_attn.kernel import tiled_attention
    m = {"tiling_mask_ok": 0.0}
    np.random.seed(42)
    q = np.random.randn(32, 16)
    k = np.random.randn(32, 16)
    v = np.random.randn(32, 16)
    try:
        out = tiled_attention(q, k, v, block_size=16, causal=False)
        if out.shape == q.shape and not np.isnan(out).any():
            m["tiling_mask_ok"] = 1.0
    except Exception:
        pass
    return m
