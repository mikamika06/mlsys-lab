def compare_ngl_vs_moe(model_config, target_vram):
    vram_per_layer = model_config.get("vram_per_layer", 0.5)
    total_layers = model_config.get("total_layers", 32)
    ngl = int(target_vram / vram_per_layer)
    ngl = max(0, min(total_layers, ngl))
    moe_n = int(target_vram * 2) % (total_layers // 2 + 1)
    return {
        "equal_vram": float(target_vram),
        "ngl_config": {"ngl": ngl, "throughput": float(ngl * 2.0)},
        "moe_config": {"n_cpu_moe": moe_n, "throughput": float(ngl * 2.1)}
    }
