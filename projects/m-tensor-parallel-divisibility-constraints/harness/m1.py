import ref


def check(workdir):
    from tp_marlin.analyze import check_eligibility, evaluate_performance

    out = {"eligibility_match": 0.0, "perf_match": 0.0}
    tp_sizes = [1, 2, 4, 8]

    for seed in [42, 43]:
        layers = ref.gen_layers(seed)
        for tp in tp_sizes:
            want_elig = ref.check_eligibility(layers, tp)
            got_elig = check_eligibility(layers, tp)
            if want_elig != got_elig:
                out["_note"] = f"eligibility mismatch on seed {seed}, tp_size {tp}"
                return out

            want_perf = ref.evaluate_performance(layers, tp)
            got_perf = evaluate_performance(layers, tp)
            if want_perf["eligible_count"] != got_perf.get("eligible_count"):
                out["_note"] = "performance dict eligible_count mismatch"
                return out

            diff = abs(want_perf["estimated_time"] - got_perf.get("estimated_time", -1.0))
            if diff > 1e-5:
                out["_note"] = f"performance estimated_time mismatch. expected {want_perf['estimated_time']}, got {got_perf.get('estimated_time')}"
                return out

    out["eligibility_match"] = 1.0
    out["perf_match"] = 1.0
    return out
