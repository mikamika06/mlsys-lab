def compute_scaling_efficiency(configs, throughputs):
    efficiencies = []
    for cfg, tp in zip(configs, throughputs):
        base_instances = sum(ig.get("count", 1) for ig in cfg.get("instance_group", []))
        ideal = base_instances * throughputs[0] / max(sum(ig.get("count", 1) for ig in configs[0].get("instance_group", [])), 1)
        eff = tp / ideal if ideal > 0 else 0.0
        efficiencies.append(eff)
    return efficiencies
