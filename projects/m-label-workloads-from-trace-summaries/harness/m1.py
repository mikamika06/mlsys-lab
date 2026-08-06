import ref


def check(workdir):
    from diag.labels import label_workloads

    out = {"traces_matched": 0.0, "total_traces": float(len(ref.TRACES))}
    ok = 0
    for i, trace in enumerate(ref.TRACES):
        want = ref.label_workloads(trace)
        got = label_workloads(trace)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"trace {i}: got {got}, reference {want}"
    out["traces_matched"] = float(ok)
    return out
