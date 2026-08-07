import numpy as np
import ref


def check(workdir):
    from splitkv.combine import split_kv_attention

    out = {"max_rel_err": 1.0, "exactness_passed": 0.0}
    rng = np.random.RandomState(42)
    max_err = 0.0

    shapes = [
        (1, 4, 1, 64, 512, 4),
        (2, 8, 1, 128, 2048, 8),
        (1, 1, 1, 32, 256, 2),
        (4, 2, 1, 64, 1024, 16),
    ]

    all_ok = True
    for b, h, q_l, d, kv_l, splits in shapes:
        q = rng.randn(b, h, q_l, d)
        k = rng.randn(b, h, kv_l, d)
        v = rng.randn(b, h, kv_l, d)

        ref_out, ref_lse = ref.naive_attention(q, k, v)
        got_out, got_lse = split_kv_attention(q, k, v, splits)

        err_out = np.max(np.abs(got_out - ref_out) / (np.abs(ref_out) + 1e-8))
        err_lse = np.max(np.abs(got_lse - ref_lse) / (np.abs(ref_lse) + 1e-8))
        err = max(float(err_out), float(err_lse))

        if err > max_err:
            max_err = err

        if err > 1e-4:
            all_ok = False
            if "_note" not in out:
                out["_note"] = f"shape ({b},{h},{kv_l}) splits={splits}: err={err:.6e}"

    out["max_rel_err"] = float(max_err)
    if all_ok and max_err <= 1e-4:
        out["exactness_passed"] = 1.0

    return out
