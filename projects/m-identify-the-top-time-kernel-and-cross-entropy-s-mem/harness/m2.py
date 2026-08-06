import ref


def check(workdir):
    from profalyze.memory import cross_entropy_memory_share

    out = {"memory_share_match": 0.0}
    trace = ref.TRACES[0]
    want = ref.cross_entropy_memory_share(trace)
    got = cross_entropy_memory_share(trace)
    if abs(got - want) < 1e-5:
        out["memory_share_match"] = 1.0
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
