import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"rel_err": 1.0}
    try:
        from attnbwd.backward import attention_backward, generate_dropout_mask
    except Exception as e:
        out["_note"] = f"Failed to import from attnbwd.backward: {e}"
        return out

    shapes = [
        (1, 2, 8, 16),
        (2, 4, 16, 32),
        (2, 2, 32, 16),
    ]
    params = [
        (0.0, 42),
        (0.1, 101),
        (0.25, 999),
    ]

    max_err = 0.0
    try:
        for B, H, N, D in shapes:
            rng = np.random.RandomState(42)
            Q = rng.randn(B, H, N, D)
            K = rng.randn(B, H, N, D)
            V = rng.randn(B, H, N, D)
            dO = rng.randn(B, H, N, D)

            for p, seed in params:
                mask_want = ref.generate_dropout_mask((B, H, N, N), p, seed)
                mask_got = generate_dropout_mask((B, H, N, N), p, seed)
                err_m = np.max(np.abs(mask_got - mask_want))
                max_err = max(max_err, float(err_m))

                dQ_w, dK_w, dV_w = ref.attention_backward(Q, K, V, dO, p, seed)
                dQ_g, dK_g, dV_g = attention_backward(Q, K, V, dO, p, seed)

                for g, w in [(dQ_g, dQ_w), (dK_g, dK_w), (dV_g, dV_w)]:
                    err = np.max(np.abs(g - w) / (np.abs(w) + 1e-8))
                    max_err = max(max_err, float(err))
    except Exception as e:
        out["_note"] = f"Execution error: {e}"
        return out

    out["rel_err"] = float(max_err)
    return out
