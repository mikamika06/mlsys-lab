def count_mlp_kernels(fusion_configs):
    kernels = set()
    for cfg in fusion_configs:
        fused = cfg.get("fused", False)
        layers = cfg.get("layers", 1)
        if fused:
            kernels.add(layers)
        else:
            kernels.add(layers * 2)
    return len(kernels) + len(fusion_configs)
