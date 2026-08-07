import ref


def check(workdir):
    from batching.coalesce import verify_coalescing
    out = {"coalesce_matched": 0.0}
    ok = 0
    for s in ref.SIZES:
        for t in ref.TIMEOUTS:
            want = ref.verify_coalescing(ref.REQUESTS, s, t)
            try:
                got = verify_coalescing(ref.REQUESTS, s, t)
                if got == want:
                    ok += 1
            except Exception:
                pass
    out["coalesce_matched"] = float(ok)
    return out
