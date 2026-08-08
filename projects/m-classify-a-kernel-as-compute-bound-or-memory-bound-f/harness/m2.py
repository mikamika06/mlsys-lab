import ref


def check(workdir):
    from kernelstats.analyzer import analyze_trace

    out = {"classifications_matched": 0.0}
    spec = ref.HARDWARE_SPECS[0]
    oracle = ref.get_oracle_data()
    match_count = 0
    for i, trace in enumerate(ref.TRACES):
        res = analyze_trace(trace, spec)
        want_class = oracle[i]["classification"]
        got_class = res.get("classification")
        if got_class == want_class:
            match_count += 1
        elif "_note" not in out:
            out["_note"] = f"trace {i}: got {got_class}, want {want_class}"

    if match_count == len(ref.TRACES):
        out["classifications_matched"] = 1.0
    return out
