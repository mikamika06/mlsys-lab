import ref

def check(workdir):
    from ring.sweep import bucket_cap_sweep

    param_sizes = [10.0, 5.0, 20.0, 2.0, 8.0, 15.0]
    candidates = [5, 15, 30, 50]

    ref_res = ref.run_bucket_sweep(param_sizes, candidates)
    got_res = bucket_cap_sweep(param_sizes, candidates)

    out = {"optimal_bucket_match": 0.0, "sweep_count_match": 0.0}

    if got_res is None or not isinstance(got_res, dict):
        out["_note"] = "bucket_cap_sweep must return a dictionary"
        return out

    if got_res.get("best_cap") == ref_res["best_cap"]:
        out["optimal_bucket_match"] = 1.0
    else:
        out["_note"] = f"expected best_cap {ref_res['best_cap']}, got {got_res.get('best_cap')}"

    if set(got_res.get("results", {}).keys()) == set(candidates):
        out["sweep_count_match"] = 1.0
    else:
        out["_note"] = "sweep results dictionary keys do not match candidates"

    return out
