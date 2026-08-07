import ref


def check(workdir):
    from ortperf.agg import aggregate_op_types
    out = {"agg_matched": 0.0}
    ok = 0
    for i, p in enumerate(ref.PROFILES):
        want = ref.aggregate_op_types(p)
        got = aggregate_op_types(p)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"profile {i}: got {got}, want {want}"
    if ok == len(ref.PROFILES):
        out["agg_matched"] = 1.0
    return out
