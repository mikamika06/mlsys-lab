import ref

def check(workdir):
    from radix.cache import FlatCache, RadixCache
    fc = FlatCache(capacity=3)
    rc = RadixCache(capacity=3)
    fc_hits = 0
    rc_hits = 0
    for _ in range(3):
        for trace in ref.TRACES:
            if fc.access(trace):
                fc_hits += 1
            if rc.access(trace):
                rc_hits += 1
    if rc_hits >= fc_hits:
        return {"radix_beats_flat": 1.0}
    return {"radix_beats_flat": 0.0, "_note": f"rc_hits {rc_hits} vs fc_hits {fc_hits}"}
