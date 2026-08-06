import ref

def check(workdir):
    from blockhash.hashing import block_hashes

    out = {"hashes_match": 0.0}
    ok = True
    for i, (t1, t2, bs) in enumerate(ref.FIXTURES):
        want = ref.block_hashes(t1, bs)
        got = block_hashes(t1, bs)
        if want != got:
            ok = False
            out["_note"] = f"fixture {i}: want {want[:2]}, got {got[:2]}"
            break

    if ok:
        out["hashes_match"] = 1.0
    return out
