import ref


def check(workdir):
    from latmetrics.decomposition import calculate_percentile, decompose_latencies

    out = {"percentile_rel_err": 1.0, "decomposition_matches": 0.0}

    max_rel_err = 0.0
    data = [r["queue_ms"] for r in ref.SAMPLE_REQUESTS]
    for p in [50.0, 90.0, 95.0, 99.0]:
        for method in ["nearest", "linear"]:
            want = ref.calculate_percentile(data, p, method=method)
            got = calculate_percentile(data, p, method=method)
            err = abs(got - want) / max(1e-9, abs(want))
            if err > max_rel_err:
                max_rel_err = err

    out["percentile_rel_err"] = float(max_rel_err)

    decomp_ok = True
    for method in ["nearest", "linear"]:
        want_d = ref.decompose_latencies(ref.SAMPLE_REQUESTS, method=method)
        got_d = decompose_latencies(ref.SAMPLE_REQUESTS, method=method)
        for k, want_v in want_d.items():
            got_v = got_d.get(k, 0.0)
            if abs(got_v - want_v) > 1e-5:
                decomp_ok = False
                out["_note"] = f"mismatch for {k} with method={method}: got {got_v}, want {want_v}"
                break
        if not decomp_ok:
            break

    if decomp_ok:
        out["decomposition_matches"] = 1.0

    return out
