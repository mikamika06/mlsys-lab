def compute_p99_ttft(family: dict, target_quantile: float = 0.99) -> dict:
    """Compute quantile TTFT and linear interpolation error bound."""
    results = {}
    samples = family.get("samples", [])
    groups = {}
    for s in samples:
        if not s["name"].endswith("_bucket"):
            continue
        labels = s["labels"].copy()
        if "le" not in labels:
            continue
        le_str = labels.pop("le")
        try:
            le_val = float(le_str)
        except ValueError:
            continue
        key = frozenset(labels.items())
        if key not in groups:
            groups[key] = []
        groups[key].append((le_val, s["value"]))
    for key, buckets in groups.items():
        buckets.sort(key=lambda x: x[0])
        if not buckets:
            results[key] = (0.0, 0.0)
            continue
        total_count = buckets[-1][1]
        if total_count <= 0:
            results[key] = (0.0, 0.0)
            continue
        target_count = target_quantile * total_count
        prev_le = 0.0
        prev_count = 0.0
        p99_val = buckets[-1][0]
        err_bound = 0.0
        for le, count in buckets:
            if count >= target_count:
                if count > prev_count:
                    frac = (target_count - prev_count) / (count - prev_count)
                    p99_val = prev_le + frac * (le - prev_le)
                else:
                    p99_val = prev_le
                err_bound = le - prev_le
                break
            prev_le = le
            prev_count = count
        results[key] = (p99_val, err_bound)
    return results
