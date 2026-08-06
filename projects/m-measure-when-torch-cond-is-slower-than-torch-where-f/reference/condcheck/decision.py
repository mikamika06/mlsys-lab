from condcheck.measurement import compute_latency_ratio


def decide_branch_strategy(case):
    ratio = compute_latency_ratio(case["ops"], case["tensor_elements"])
    return "cond" if ratio > 1.0 else "where"
