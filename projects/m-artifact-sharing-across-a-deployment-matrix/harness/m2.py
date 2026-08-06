import ref


def check(workdir):
    out = {"instances_matched": 0.0, "slo_satisfied": 0.0}
    try:
        from matrix.slo import calculate_required_instances
    except Exception as e:
        out["_note"] = f"Failed to import calculate_required_instances: {e}"
        return out

    test_cases = [
        (500.0, 10.0, 25.0, 35.0, 0.80),
        (1200.0, 15.0, 30.0, 40.0, 0.85),
        (250.0, 50.0, 90.0, 120.0, 0.75),
    ]

    matched = 0
    slo_ok = 0
    for arrival, mean_st, p99_st, target_slo, max_util in test_cases:
        expected = ref.ref_calculate_instances(arrival, mean_st, p99_st, target_slo, max_util)
        got = calculate_required_instances(arrival, mean_st, p99_st, target_slo, max_util)

        if got == expected:
            matched += 1

        service_rate = 1000.0 / mean_st
        total_cap = got * service_rate
        rho = arrival / total_cap
        queue_wait = (rho / (1.0 - rho)) * mean_st / got if rho < 1.0 else 9999.0
        est_p99 = p99_st + 2.326 * queue_wait

        if est_p99 <= target_slo and rho <= max_util:
            slo_ok += 1

    if matched == len(test_cases):
        out["instances_matched"] = 1.0
    if slo_ok == len(test_cases):
        out["slo_satisfied"] = 1.0

    return out
