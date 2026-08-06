def find_optimal_eagle_config(configs: list, kv_cache_budget_bytes: int) -> dict:
    best_cfg = None
    best_score = -1.0
    for cfg in configs:
        cost = cfg.get("kv_bytes", float("inf"))
        if cost <= kv_cache_budget_bytes:
            score = cfg.get("throughput_score", 0.0)
            if score > best_score:
                best_score = score
                best_cfg = cfg
    return best_cfg
