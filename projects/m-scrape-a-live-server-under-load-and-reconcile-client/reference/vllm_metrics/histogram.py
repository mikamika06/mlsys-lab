def reconstruct_histogram(metric_lines):
    buckets = []
    count = 0.0
    sum_val = 0.0
    for labels, val in metric_lines:
        if "le" in labels:
            le_str = labels["le"]
            if le_str == "+Inf":
                le = float("inf")
            else:
                le = float(le_str)
            buckets.append((le, val))
        elif labels.get("__name__", "").endswith("_count") or "vllm:request_latency_seconds_count" in str(metric_lines):
            pass
    buckets.sort(key=lambda x: x[0])
    return buckets
