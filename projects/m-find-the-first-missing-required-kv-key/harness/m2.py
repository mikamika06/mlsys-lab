import ref


def check(workdir):
    from ggufschema.parser import derive_gqa_and_head_dim

    out = {"params_matched": 0.0}
    ok = 0
    for i, meta in enumerate(ref.METADATAS):
        def local_ref(m):
            hd = m.get("embedding_length", 0) // m.get("attention.head_count", 1)
            hc = m.get("attention.head_count", 0)
            hckv = m.get("attention.head_count_kv", hc)
            return {"gqa_ratio": int(hc // hckv if hckv > 0 else 1), "head_dim": int(hd)}
        w = local_ref(meta)
        got = derive_gqa_and_head_dim(meta)
        if got == w:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"metadata {i}: got {got}, want {w}"
    out["params_matched"] = float(ok)
    return out
