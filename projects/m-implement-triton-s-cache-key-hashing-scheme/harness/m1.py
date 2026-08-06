import ref


def check(workdir):
    from tcache.hashing import compute_cache_key

    out = {"hashes_matched": 0.0}
    ok = 0
    for req in ref.REQUESTS:
        want = ref.compute_cache_key(req)
        got = compute_cache_key(req)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"request {req}: got {got}, want {want}"
    out["hashes_matched"] = 1.0 if ok == len(ref.REQUESTS) else 0.0
    return out
