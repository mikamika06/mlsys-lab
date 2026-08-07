import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from engine_diag.cudagraph import compute_padded_tokens, optimize_buckets
    except ImportError as e:
        return {
            "waste_matches": 0.0,
            "optimal_buckets_match": 0.0,
            "_note": f"Import error: {e}"
        }

    out = {"waste_matches": 0.0, "optimal_buckets_match": 0.0}

    waste_ok = True
    buckets_ok = True

    for i, tc in enumerate(ref.BUCKET_TEST_CASES):
        k = tc["k"]
        max_batch = tc["max_batch"]
        hist = tc["histogram"]

        test_buckets = [max_batch // 2, max_batch]
        want_waste = ref.compute_padded_tokens(test_buckets, hist)
        try:
            got_waste = compute_padded_tokens(test_buckets, hist)
            if got_waste != want_waste:
                waste_ok = False
                out["_note"] = f"Waste mismatch TC {i}: got {got_waste}, want {want_waste}"
        except Exception as e:
            waste_ok = False
            out["_note"] = f"Error computing waste TC {i}: {e}"

        want_opt = ref.optimize_buckets(k, max_batch, hist)
        try:
            got_opt = optimize_buckets(k, max_batch, hist)
            got_opt_waste = ref.compute_padded_tokens(got_opt, hist)
            want_opt_waste = ref.compute_padded_tokens(want_opt, hist)

            if got_opt_waste != want_opt_waste or len(got_opt) != k or max_batch not in got_opt:
                buckets_ok = False
                out["_note"] = f"Optimization mismatch TC {i}: got {got_opt}, want {want_opt}"
        except Exception as e:
            buckets_ok = False
            out["_note"] = f"Error optimizing buckets TC {i}: {e}"

    out["waste_matches"] = 1.0 if waste_ok else 0.0
    out["optimal_buckets_match"] = 1.0 if buckets_ok else 0.0
    return out
