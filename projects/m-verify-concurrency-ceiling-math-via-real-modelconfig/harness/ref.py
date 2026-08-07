CONFIGS = [
    {"max_batch_size": 8, "instance_group": [{"count": 1}], "dynamic_range_limit": 4},
    {"max_batch_size": 4, "instance_group": [{"count": 2}], "dynamic_range_limit": 2},
    {"max_batch_size": 16, "instance_group": [{"count": 4}], "dynamic_range_limit": 1}
]

def compute_concurrency_ceiling(config):
    max_batch = config.get("max_batch_size", 1)
    instances = config.get("instance_group", [])
    total_instances = sum(ig.get("count", 1) for ig in instances)
    dynamic_ranges = config.get("dynamic_range_limit", max_batch)
    ceiling = max_batch * max(total_instances, 1) * dynamic_ranges
    return ceiling

def compute_scaling_efficiency(configs, throughputs):
    efficiencies = []
    for cfg, tp in zip(configs, throughputs):
        base_instances = sum(ig.get("count", 1) for ig in cfg.get("instance_group", []))
        ideal = base_instances * throughputs[0] / max(sum(ig.get("count", 1) for ig in configs[0].get("instance_group", [])), 1)
        eff = tp / ideal if ideal > 0 else 0.0
        efficiencies.append(eff)
    return efficiencies

def classify_error(error_string):
    s = error_string.lower()
    if "dynamic shape" in s or "shape mismatch" in s:
        return "DYNAMIC_SHAPE_ERROR"
    if "out of memory" in s or "cuda oom" in s:
        return "OOM_ERROR"
    if "concurrency" in s or "queue overflow" in s:
        return "CONCURRENCY_CEILING_EXCEEDED"
    return "UNKNOWN_ERROR"
