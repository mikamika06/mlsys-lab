import numpy as np
import ref


def check(workdir):
    from fused_attn.kernel import tiled_attention
    passed = 0
    lengths = [32, 64, 128]
    for length in lengths:
        np.random.seed(length)
        q = np.random.randn(length, 16)
        k = np.random.randn(length, 16)
        v = np.random.randn(length, 16)
        try:
            ref_out = ref.compute_reference_attention(q, k, v, causal=False)
            out = tiled_attention(q, k, v, block_size=16, causal=False)
            if np.allclose(ref_out, out, atol=1e-4, rtol=1e-4):
                passed += 1
        except Exception:
            pass
    return {"lengths_passed": float(passed)}
