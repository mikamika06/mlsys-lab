import ref


def check(workdir):
    from autotune_metrics.analyzer import compare_latencies
    cases = ref.generate_cases(seed=123)
    matched_lat = 0
    matched_comp = 0
    total = len(cases)

    for case in cases:
        got = compare_latencies(case["default_record"], case["autotune_record"])
        if got.get("latency_ratio") == case["expected"]["latency_ratio"]:
            matched_lat += 1
        if got.get("compile_time_ratio") == case["expected"]["compile_time_ratio"]:
            matched_comp += 1

    out = {
        "latency_ratio_match": 1.0 if matched_lat == total else 0.0,
        "compile_time_match": 1.0 if matched_comp == total else 0.0
    }
    if matched_lat != total:
        out["_note"] = f"latency ratio mismatched on {total - matched_lat} cases"
    return out
