def simulate_oom(case):
    model_sz = case["model_size_bytes"]
    kv_rate = case["kv_bytes_per_token"]
    limit = case["memory_limit_bytes"]
    results = []
    for ctx in case["context_lengths"]:
        total = model_sz + ctx * kv_rate
        oom = total > limit
        results.append({"context": ctx, "total_bytes": total, "oom": oom})
    return results
