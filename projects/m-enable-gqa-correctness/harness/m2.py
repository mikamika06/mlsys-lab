import numpy as np

import ref


def _close(a, b, tol=1e-4):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.max(np.abs(a - b)) <= tol


def check(workdir):
    from gqa import repeat_kv, scaled_dot_product_attention

    out = {"expand_match": 0.0, "no_op_match": 0.0, "causal_gqa_match": 0.0}
    note = None

    expand_ok = True
    for i, (shape, n_rep) in enumerate([((2, 2, 3, 4), 3), ((1, 3, 5, 2), 4), ((4, 1, 2, 6), 5)]):
        rng = np.random.default_rng(100 + i)
        x = rng.standard_normal(shape)
        want = ref.repeat_kv(x, n_rep)
        try:
            got = np.asarray(repeat_kv(x, n_rep))
        except Exception as e:  # noqa: BLE001
            expand_ok = False
            note = note or f"repeat_kv{shape},{n_rep}: {type(e).__name__}: {str(e)[:120]}"
            continue
        if got.shape != want.shape or not np.array_equal(got, want):
            expand_ok = False
            note = note or f"repeat_kv{shape},{n_rep}: does not match the oracle expansion"
    out["expand_match"] = 1.0 if expand_ok else 0.0

    case = {"B": 2, "Hq": 5, "Hkv": 5, "L": 5, "S": 5, "D": 8, "seed": 42}
    q, k, v = ref.make_qkv(case)
    a = scaled_dot_product_attention(q, k, v, enable_gqa=True)
    b = scaled_dot_product_attention(q, k, v, enable_gqa=False)
    ok = _close(a, b, 1e-8)
    out["no_op_match"] = 1.0 if ok else 0.0
    if not ok and note is None:
        note = "enable_gqa changes the result even when q_heads == kv_heads"

    causal_ok = True
    for i, ccase in enumerate([c for c in ref.CASES if c["is_causal"]]):
        q, k, v = ref.make_qkv(ccase)
        want = ref.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        got = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        if not _close(got, want):
            causal_ok = False
            note = note or f"causal case {i}: does not match the oracle under enable_gqa"
    out["causal_gqa_match"] = 1.0 if causal_ok else 0.0

    if note:
        out["_note"] = note
    return out
