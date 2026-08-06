import ref


def check(workdir):
    from servermon.metrics import parse_trace
    out = {"traces_parsed": 0.0}
    ok = 0
    for i, t in enumerate(ref.TRACES):
        got = parse_trace(t)
        want = ref.load_traces([t])[0]
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"trace {i}: got {got}, want {want}"
    out["traces_parsed"] = float(ok)
    return out
