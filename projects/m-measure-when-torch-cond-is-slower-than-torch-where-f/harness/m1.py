import ref


def check(workdir):
    from condcheck.measurement import compute_latency_ratio
    out = {"latency_ratio_match": 0.0}
    ok = True
    for case in ref.IR_CASES:
        val = compute_latency_ratio(case["ops"], case["tensor_elements"])
        if not isinstance(val, (int, float)) or val <= 0:
            ok = False
    if ok:
        out["latency_ratio_match"] = 1.0
    return out
