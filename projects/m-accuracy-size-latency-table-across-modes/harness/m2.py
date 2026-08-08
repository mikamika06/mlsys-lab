import ref


def check(workdir):
    from modetbl.metrics import compute_size_ratios, evaluate_tradeoffs
    out = {"size_ratio_match": 0.0, "valid_tradeoffs": 0.0}
    try:
        profiles = ref.build_profiles(ref.RAW_RECORDS)
        got_ratios = compute_size_ratios(profiles)
        want_ratios = ref.compute_size_ratios(profiles)
        if got_ratios == want_ratios:
            out["size_ratio_match"] = 1.0

        got_eval = evaluate_tradeoffs(profiles)
        want_eval = ref.evaluate_tradeoffs(profiles)
        if got_eval == want_eval:
            out["valid_tradeoffs"] = 1.0
    except Exception as e:
        out["_note"] = f"error: {type(e).__name__}: {str(e)}"
    return out
