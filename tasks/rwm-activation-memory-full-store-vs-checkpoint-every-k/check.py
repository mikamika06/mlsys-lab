def _ref(L, S, H, k):
    import numpy as np
    full = int(np.int64(L * S * H))
    checkpoints = int(np.ceil(L / k) + 1)
    ckpt = int(checkpoints * S * H + k * S * H)
    return (full, ckpt)

def grade(sol, fx) -> dict:
    cases = [
        (10, 32, 128, 5),
        (11, 16, 64, 3),
        (20, 8, 256, 4),
        (7, 1, 512, 2),
        (15, 50, 100, 7)
    ]
    ok = 1.0
    for L, S, H, k in cases:
        try:
            got = sol.compute_peak_activation_memory(L, S, H, k)
            if not isinstance(got, tuple) or len(got) != 2:
                ok = 0.0
                break
            got_full, got_ckpt = got
            ref_full, ref_ckpt = _ref(L, S, H, k)
            if (got_full, got_ckpt) != (ref_full, ref_ckpt):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
