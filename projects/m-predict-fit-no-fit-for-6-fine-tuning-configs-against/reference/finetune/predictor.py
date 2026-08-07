def predict_fits(configs, vram_budget_bytes):
    results = []
    for cfg in configs:
        bs = cfg.get("batch_size", 1)
        seq = cfg.get("seq_len", 512)
        hidden = cfg.get("hidden_dim", 768)
        layers = cfg.get("num_layers", 12)
        grad_accum = cfg.get("grad_accum", 1)
        checkpoint = cfg.get("checkpointing", False)

        base_bytes = bs * seq * hidden * layers * 4
        if not checkpoint:
            base_bytes *= 2
        frag_factor = 1.15 if seq > 1024 else 1.05
        total_estimated = int(base_bytes * frag_factor)
        fits = total_estimated <= vram_budget_bytes
        results.append({
            "config_id": cfg.get("id"),
            "estimated_bytes": total_estimated,
            "fits": fits
        })
    return results
