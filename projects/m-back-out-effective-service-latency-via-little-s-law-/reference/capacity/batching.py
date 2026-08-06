def find_optimal_batch_size(profiles, sla_latency_ms, cost_per_node_hr):
    """
    profiles: list of dicts with 'batch_size', 'latency_ms', 'tokens_per_sec'
    """
    best_batch = None
    min_cost_per_1k = float("inf")
    
    for p in profiles:
        b = p["batch_size"]
        lat = p["latency_ms"]
        tps = p["tokens_per_sec"]
        
        if lat <= sla_latency_ms:
            cost_per_sec = cost_per_node_hr / 3600.0
            cost_per_1k = (cost_per_sec / tps) * 1000.0
            if cost_per_1k < min_cost_per_1k:
                min_cost_per_1k = cost_per_1k
                best_batch = b
                
    return {"optimal_batch_size": best_batch, "min_cost_per_1k_tokens": min_cost_per_1k}
