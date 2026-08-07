def find_largest_batch_size(configs, vram_budget):
    best_bs = 0
    for cfg in sorted(configs, key=lambda x: x.get("batch_size", 0)):
        model_mem = cfg.get("model_bytes", 0)
        opt_mem = cfg.get("optimizer_bytes", 0)
        act_mem = cfg.get("activation_bytes", 0) * cfg.get("batch_size", 1)
        overhead = cfg.get("overhead_bytes", 0)
        frag = cfg.get("fragmentation_factor", 1.1)
        total = int((model_mem + opt_mem + act_mem + overhead) * frag)
        if total <= vram_budget:
            best_bs = max(best_bs, cfg.get("batch_size", 0))
    return best_bs
