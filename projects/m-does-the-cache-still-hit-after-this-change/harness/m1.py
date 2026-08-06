import ref


def check(workdir):
    from cacheutils.hasher import stable_hash

    out = {"hashes_matched": 0.0, "total": float(len(ref.PAYLOADS))}
    ok = 0
    for i, p in enumerate(ref.PAYLOADS):
        want = ref.compute_stable_hash(p)
        try:
            got = stable_hash(p)
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"payload {i} raised {type(e).__name__}"
            continue
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"payload {i}: got {got}, reference {want}"
    out["hashes_matched"] = float(ok)
    return out
