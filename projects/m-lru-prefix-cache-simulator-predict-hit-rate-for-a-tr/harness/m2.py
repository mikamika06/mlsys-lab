import ref

def check(workdir):
    from prefcache.eviction import reproduce_eviction_sequence
    out = {"sequences_matched": 0.0}
    ok = 0
    for ops, cap in ref.OPERATIONS_SET:
        want = ref.reproduce_eviction_sequence(ops, cap)
        try:
            got = reproduce_eviction_sequence(ops, cap)
        except Exception:
            got = []
        if got == want:
            ok += 1
    out["sequences_matched"] = float(ok)
    return out
