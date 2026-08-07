def predict_fits(configs, vram_budget):
    results = []
    for cfg in configs:
        model_mem = cfg.get("model_bytes", 0)
        opt_mem = cfg.get("optimizer_bytes", 0)
        act_mem = cfg.get("activation_bytes", 0)
        overhead = cfg.get("overhead_bytes", 0)
        frag_factor = cfg.get("fragmentation_factor", 1.1)
        total = int((model_mem + opt_mem + act_mem + overhead) * frag_factor)
        results.append({
            "config_id": cfg["config_id"],
            "fits": total <= vram_budget,
            "estimated_bytes": total
        })
    return results
