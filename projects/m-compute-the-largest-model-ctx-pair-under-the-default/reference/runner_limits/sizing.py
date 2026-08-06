def compute_largest_pair(models, ceiling, overhead=1024*1024*1024):
    best_model = None
    best_ctx = 0
    for m in models:
        base_mem = m["params"] * m["bytes_per_param"] + overhead
        if base_mem >= ceiling:
            continue
        available = ceiling - base_mem
        max_ctx = int(available // m["kv_per_token"])
        if max_ctx > 0:
            if best_model is None or m["params"] > best_model["params"] or (m["params"] == best_model["params"] and max_ctx > best_ctx):
                best_model = m["name"]
                best_ctx = max_ctx
    return best_model, best_ctx
