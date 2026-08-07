import ref


def check(workdir):
    from pt2e_counts.analyzer import compute_conversion_deltas
    out = {"deltas_match": 0.0, "overhead_ratio": 0.0}
    ok_deltas = 0
    ok_overhead = 0
    for item in ref.GRAPHS:
        orig = item["orig"]
        prep = item["prep"]
        conv = item["conv"]
        want = ref.compute_conversion_deltas(orig, prep, conv)
        got = compute_conversion_deltas(orig, prep, conv)
        if got.get("prep_delta") == want.get("prep_delta") and got.get("conv_delta") == want.get("conv_delta"):
            ok_deltas += 1
        want_orig_total = want["orig"].get("total", 0)
        want_prep_total = want["prep"].get("total", 0)
        got_orig_total = got["orig"].get("total", 0)
        got_prep_total = got["prep"].get("total", 0)
        if want_orig_total > 0 and got_orig_total > 0:
            if abs((got_prep_total / got_orig_total) - (want_prep_total / want_orig_total)) < 1e-5:
                ok_overhead += 1

    out["deltas_match"] = 1.0 if ok_deltas == len(ref.GRAPHS) else 0.0
    out["overhead_ratio"] = 1.0 if ok_overhead == len(ref.GRAPHS) else 0.0
    return out
