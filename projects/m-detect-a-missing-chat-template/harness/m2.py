import ref


def check(workdir):
    from calib.optimizer import optimize_calibration_params

    out = {"optimal_match": 0.0, "compute_bound": 0.0}
    matched = 0
    bound_ok = 0

    for test in ref.OPTIMIZATION_TESTS:
        got = optimize_calibration_params(
            test["target_tokens"], test["max_seqlen"], test["available_memory_mb"]
        )
        if isinstance(got, dict) and "n" in got and "seqlen" in got:
            matched += 1
            if got["n"] > 0 and got["seqlen"] > 0:
                bound_ok += 1

    out["optimal_match"] = 1.0 if matched == len(ref.OPTIMIZATION_TESTS) else 0.0
    out["compute_bound"] = 1.0 if bound_ok == len(ref.OPTIMIZATION_TESTS) else 0.0
    return out
