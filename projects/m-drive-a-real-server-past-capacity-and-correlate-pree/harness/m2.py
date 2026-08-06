import ref


def check(workdir):
    from servermon.correlate import compute_p99_correlation
    out = {"correlation_match": 0.0, "p99_accuracy": 0.0}
    ok_corr = 0
    ok_p99 = 0
    for i, t in enumerate(ref.TRACES):
        res = ref.analyze_server(t)
        got = compute_p99_correlation(t["preemptions"], t["latencies"])
        if abs(got.get("correlation", 0) - res["correlation"]) < 1e-3:
            ok_corr += 1
        if abs(got.get("p99", 0) - res["p99"]) < 1e-3:
            ok_p99 += 1
    out["correlation_match"] = 1.0 if ok_corr == len(ref.TRACES) else 0.0
    out["p99_accuracy"] = 1.0 if ok_p99 == len(ref.TRACES) else 0.0
    return out
