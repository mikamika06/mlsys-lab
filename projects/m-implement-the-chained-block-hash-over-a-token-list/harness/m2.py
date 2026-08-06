import ref

def check(workdir):
    from blockhash.hashing import block_hashes, reusable_blocks, divergence

    out = {"reusable_match": 0.0, "divergence_match": 0.0}
    re_ok = True
    div_ok = True

    for i, (t1, t2, bs) in enumerate(ref.FIXTURES):
        h1 = block_hashes(t1, bs)
        h2 = block_hashes(t2, bs)

        want_re = ref.reusable_blocks(h1, h2)
        got_re = reusable_blocks(h1, h2)
        if want_re != got_re:
            re_ok = False
            out.setdefault("_note", "")
            out["_note"] += f"reusable mismatch fixture {i}: want {want_re}, got {got_re}. "

        want_div = ref.divergence(t1, t2, bs)
        got_div = divergence(t1, t2, bs)
        if want_div != got_div:
            div_ok = False
            out.setdefault("_note", "")
            out["_note"] += f"divergence mismatch fixture {i}: want {want_div}, got {got_div}. "

    if re_ok:
        out["reusable_match"] = 1.0
    if div_ok:
        out["divergence_match"] = 1.0

    return out
