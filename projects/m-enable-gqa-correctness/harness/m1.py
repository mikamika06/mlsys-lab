import numpy as np

import ref


def check(workdir):
    from gqa import scaled_dot_product_attention

    out = {"max_abs_err": 0.0}
    worst = 0.0
    note = None
    for i, case in enumerate(ref.CASES):
        q, k, v = ref.make_qkv(case)
        want = ref.scaled_dot_product_attention(q, k, v, is_causal=case["is_causal"], enable_gqa=True)
        try:
            got = np.asarray(
                scaled_dot_product_attention(q, k, v, is_causal=case["is_causal"], enable_gqa=True),
                dtype=np.float64,
            )
        except Exception as e:  # noqa: BLE001
            worst = max(worst, 1e9)
            note = note or f"case {i}: {type(e).__name__}: {str(e)[:120]}"
            continue
        if got.shape != want.shape:
            worst = max(worst, 1e9)
            note = note or f"case {i}: shape {got.shape} != expected {want.shape}"
            continue
        err = float(np.max(np.abs(got - want)))
        worst = max(worst, err)
        if note is None and err > 1e-4:
            note = f"case {i}: max abs err {err:.3g}"
    out["max_abs_err"] = worst
    if note:
        out["_note"] = note
    return out
