def optimal_eagle_config(kv_budget_bytes: int, configs: list) -> dict:
    best = None
    max_score = -1.0
    for cfg in configs:
        if cfg["kv_bytes"] <= kv_budget_bytes:
            score = cfg["throughput"]
            if score > max_score:
                max_score = score
                best = cfg
    if best is None:
        return configs[0] if configs else {}
    return best
