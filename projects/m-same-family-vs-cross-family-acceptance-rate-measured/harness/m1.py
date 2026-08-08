import ref


def check(workdir):
    from speculative.metrics import compute_acceptance_rate

    out = {"acceptance_rate_match": 0.0, "_note": ""}
    ok = 0
    for i, trace in enumerate(ref.TRACES):
        want = ref.compute_acceptance_rate(trace)
        got = compute_acceptance_rate(trace)
        if abs(want - got) < 1e-5:
            ok += 1
        elif not out["_note"]:
            out["_note"] = f"trace {i}: got {got}, want {want}"
    if ok == len(ref.TRACES):
        out["acceptance_rate_match"] = 1.0
    return out
