def compute_largest_pair(models, contexts, ceiling_bytes):
    best_m = None
    best_c = None
    max_score = -1
    for m in models:
        weights = m.get("weight_bytes", 0)
        kv_per_token = m.get("kv_bytes_per_token", 0)
        overhead = m.get("overhead_bytes", 0)
        for c in contexts:
            total = weights + (kv_per_token * c) + overhead
            if total <= ceiling_bytes:
                score = weights + c
                if score > max_score:
                    max_score = score
                    best_m = m["name"]
                    best_c = c
    return (best_m, best_c)
