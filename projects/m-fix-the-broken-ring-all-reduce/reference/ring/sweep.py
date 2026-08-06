def bucket_cap_sweep(param_sizes_mb, candidate_caps):
    results = {}
    for cap in candidate_caps:
        buckets = []
        cur = 0
        for p in param_sizes_mb:
            if cur + p > cap and cur > 0:
                buckets.append(cur)
                cur = p
            else:
                cur += p
        if cur > 0:
            buckets.append(cur)
        score = sum(b * 0.95 for b in buckets) + len(buckets) * 0.1
        results[cap] = score
    best_cap = min(results, key=results.get)
    return {"best_cap": best_cap, "results": results}
