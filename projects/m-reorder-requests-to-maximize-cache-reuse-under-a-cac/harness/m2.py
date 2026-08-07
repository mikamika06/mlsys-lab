import ref


def check(workdir):
    from cacheopt.analysis import identify_eviction

    out = {"eviction_matched": 0.0}
    reqs = ref.CONFIGS[0]
    want = ref.identify_eviction(reqs, 3)
    got = identify_eviction(reqs, 3)
    if got == want:
        out["eviction_matched"] = 1.0
    else:
        out["_note"] = f"got eviction {got}, want {want}"
    return out
