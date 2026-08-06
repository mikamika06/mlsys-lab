import ref


def check(workdir):
    from auditor.budget import predict_ready_time

    out = {"budget_matched": 0.0}
    ok = 0
    for case in ref.BUDGET_CASES:
        want = ref.predict_ready_time(
            case["pull_size"], case["weight_size"],
            case["compile_factor"], case["pull_speed"], case["weight_speed"]
        )
        got = predict_ready_time(
            case["pull_size"], case["weight_size"],
            case["compile_factor"], case["pull_speed"], case["weight_speed"]
        )
        if abs(got - want) < 1e-5:
            ok += 1
    if ok == len(ref.BUDGET_CASES):
        out["budget_matched"] = 1.0
    return out
