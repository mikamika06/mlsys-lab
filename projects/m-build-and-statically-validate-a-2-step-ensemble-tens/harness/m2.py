import ref


def check(workdir):
    from ensemble.latency import compute_ensemble_latency

    out = {"latency_matched": 0.0}
    ok = 0
    for i, data in enumerate(ref.LATENCY_DATA):
        want = ref.compute_latencies(data)
        got = compute_ensemble_latency(data["step1_latencies"], data["step2_latencies"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"data {i}: got {got}, reference {want}"
    if ok == len(ref.LATENCY_DATA):
        out["latency_matched"] = 1.0
    return out
