def cheapest_config(model, gpus, ctx, budget):
    kv_bytes = model["num_layers"] * model["num_kv_heads"] * model["head_dim"] * 4
    req_gb = model["weights_gb"] + (kv_bytes * ctx) / (1024**3)

    best = None
    best_cost = float('inf')
    best_mem = -1

    for g in gpus:
        for n in range(1, g["count"] + 1):
            mem = n * g["mem_gb"]
            cost = n * g["cost_per_hr"]
            if mem >= req_gb and cost <= budget:
                if cost < best_cost or (cost == best_cost and mem > best_mem):
                    best = (g["name"], n)
                    best_cost = cost
                    best_mem = mem
    return best
