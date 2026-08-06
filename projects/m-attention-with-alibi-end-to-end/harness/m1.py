import numpy as np
import ref

def check(workdir):
    from alibi_attn.attention import compute_alibi_attention

    np.random.seed(42)
    shapes = [
        (2, 4, 16, 32),
        (1, 8, 32, 64),
        (2, 12, 64, 32)
    ]

    max_err = 0.0
    for b, h, seq, d in shapes:
        q = np.random.randn(b, h, seq, d).astype(np.float32)
        k = np.random.randn(b, h, seq, d).astype(np.float32)
        v = np.random.randn(b, h, seq, d).astype(np.float32)

        for softcap in [None, 10.0, 15.0]:
            want = ref.alibi_attention(q, k, v, softcap=softcap)
            got = compute_alibi_attention(q, k, v, softcap=softcap)
            err = float(np.max(np.abs(want - got)))
            if err > max_err:
                max_err = err

    return {"max_abs_err": max_err}
