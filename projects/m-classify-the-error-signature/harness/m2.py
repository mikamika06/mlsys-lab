import ref


def check(workdir):
    from errorclass.tolerance import evaluate_tolerance

    policy = {"atol": 1e-4, "rtol": 1e-3}

    stats_good = {"max_abs_diff": 1e-6, "mean_rel_diff": 1e-5, "has_nan": False}
    stats_bad = {"max_abs_diff": 0.5, "mean_rel_diff": 0.2, "has_nan": False}
    stats_nan = {"max_abs_diff": 0.1, "mean_rel_diff": 0.01, "has_nan": True}

    res_good = evaluate_tolerance(stats_good, policy)
    res_bad = evaluate_tolerance(stats_bad, policy)
    res_nan = evaluate_tolerance(stats_nan, policy)

    passed = (
        res_good.get("accepted") is True and
        res_bad.get("accepted") is False and
        res_nan.get("accepted") is False
    )

    out = {"policy_evaluated": 1.0 if passed else 0.0}
    if not passed:
        out["_note"] = f"tolerance evaluation failed logic check: good={res_good}, bad={res_bad}, nan={res_nan}"
    return out
