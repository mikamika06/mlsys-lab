def optimize_under_budget(layer_configs, memory_budget_bytes):
    selected = []
    current_memory = 0
    for cfg in sorted(layer_configs, key=lambda x: x.get("size", 0), reverse=True):
        cost = cfg.get("size", 0)
        if current_memory + cost <= memory_budget_bytes:
            selected.append(cfg["name"])
            current_memory += cost
    return {"selected_layers": selected, "total_memory": current_memory}
