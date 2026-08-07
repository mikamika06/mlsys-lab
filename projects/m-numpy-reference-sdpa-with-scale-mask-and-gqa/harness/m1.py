import ref
import numpy as np


def check(workdir):
    try:
        from sdpa.reference import numpy_sdpa
    except ImportError:
        return {"matches": 0.0}

    np.random.seed(42)
    B = 2
    L_q = 5
    L_kv = 5
    D = 16

    cases = [
        (4, 4, False, None, None),
        (4, 1, False, None, None),
        (8, 2, True, None, None),
        (8, 2, False, np.ones((L_q, L_kv), dtype=bool), 0.5),
    ]

    ok = 0
    for H_q, H_kv, is_causal, mask, scale in cases:
        q = np.random.randn(B, H_q, L_q, D)
        k = np.random.randn(B, H_kv, L_kv, D)
        v = np.random.randn(B, H_kv, L_kv, D)

        want = ref.numpy_sdpa(q, k, v, mask, is_causal, scale)
        try:
            got = numpy_sdpa(q, k, v, mask, is_causal, scale)
            if np.allclose(want, got, atol=1e-5):
                ok += 1
        except Exception:
            pass

    return {"matches": float(ok) / len(cases)}
