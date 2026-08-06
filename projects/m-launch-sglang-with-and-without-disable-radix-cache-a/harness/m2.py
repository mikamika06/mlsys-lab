import ref


def check(workdir):
    from sgl_utils import metrics
    out = {"ratio_match": 0.0, "latency_ratio_valid": 0.0}
    test_cases = [
        (10.0, 40.0),
        (25.5, 102.0),
        (5.0, 50.0)
    ]
    all_matched = True
    valid_range = True
    for enabled, disabled in test_cases:
        want = ref.compute_latency_ratio(enabled, disabled)
        got = metrics.compute_latency_ratio(enabled, disabled)
        if abs(got - want) > 1e-5:
            all_matched = False
        if not (0.0 < got <= 1.0):
            valid_range = False
    if all_matched:
        out["ratio_match"] = 1.0
    if valid_range:
        out["latency_ratio_valid"] = 1.0
    return out
