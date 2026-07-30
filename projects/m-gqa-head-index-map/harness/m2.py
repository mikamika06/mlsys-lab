import numpy as np
import ref


def check(workdir):
    from gqa.attention import expand_kv, gqa_attention

    out = {"expand_matched": 0.0, "attn_matched": 0.0, "causal_respected": 0.0,
           "cases": float(len(ref.CASES))}
    expand_ok = 0
    attn_ok = 0
    for i, case in enumerate(ref.CASES):
        q, k, v = ref.make_case(case)
        nq, nkv = case["num_q_heads"], case["num_kv_heads"]
        want_k = ref.expand_kv(k, nq, nkv)
        want_out = ref.attention(q, k, v, nkv)
        try:
            got_k = np.asarray(expand_kv(k, nq))
            got_out = np.asarray(gqa_attention(q, k, v, nkv))
        except Exception as e:  # noqa: BLE001
            if "_note" not in out:
                out["_note"] = f"case {i}: raised {type(e).__name__}: {str(e)[:120]}"
            continue
        if got_k.shape == want_k.shape and np.array_equal(got_k, want_k):
            expand_ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: expand_kv mismatch, shape got={got_k.shape} want={want_k.shape}"
        if got_out.shape == want_out.shape and np.allclose(got_out, want_out, atol=1e-6, rtol=1e-5):
            attn_ok += 1
        elif "_note" not in out:
            diff = float(np.max(np.abs(got_out - want_out))) if got_out.shape == want_out.shape else None
            out["_note"] = f"case {i}: attention mismatch, max_abs_err={diff}"
    out["expand_matched"] = float(expand_ok)
    out["attn_matched"] = float(attn_ok)

    causal_ok = True
    case = ref.CASES[0]
    q, k, v = ref.make_case(case)
    nq, nkv = case["num_q_heads"], case["num_kv_heads"]
    try:
        base = np.asarray(gqa_attention(q, k, v, nkv))
        k2, v2 = k.copy(), v.copy()
        k2[:, -1, :] += 100.0
        v2[:, -1, :] += 100.0
        perturbed = np.asarray(gqa_attention(q, k2, v2, nkv))
        seq_q = q.shape[1]
        causal_ok = base.shape == perturbed.shape and bool(
            np.allclose(base[:, :seq_q - 1, :], perturbed[:, :seq_q - 1, :], atol=1e-6, rtol=1e-5)
        )
    except Exception:  # noqa: BLE001
        causal_ok = False
    out["causal_respected"] = 1.0 if causal_ok else 0.0

    return out
