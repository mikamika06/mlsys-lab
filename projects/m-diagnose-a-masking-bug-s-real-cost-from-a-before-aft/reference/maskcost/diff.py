def compute_diff(baseline, modified):
    diffs = {}
    all_keys = set(baseline.keys()).union(set(modified.keys()))
    for k in all_keys:
        b = baseline.get(k, 0.0)
        m = modified.get(k, 0.0)
        if isinstance(b, (int, float)) and isinstance(m, (int, float)):
            abs_diff = m - b
            pct_diff = (abs_diff / b * 100.0) if b != 0 else 0.0
            diffs[k] = {"baseline": b, "modified": m, "abs_diff": abs_diff, "pct_diff": pct_diff}
        else:
            diffs[k] = {"baseline": b, "modified": m, "abs_diff": 0.0, "pct_diff": 0.0}

    primary_bottleneck = "none"
    max_increase = -float("inf")
    for k, v in diffs.items():
        if isinstance(v["abs_diff"], (int, float)) and v["abs_diff"] > max_increase:
            max_increase = v["abs_diff"]
            primary_bottleneck = k

    return {
        "diffs": diffs,
        "primary_bottleneck": primary_bottleneck
    }
