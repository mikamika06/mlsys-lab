def compute_concurrency_ceiling(config):
    max_batch = config.get("max_batch_size", 1)
    instances = config.get("instance_group", [])
    total_instances = sum(ig.get("count", 1) for ig in instances)
    dynamic_ranges = config.get("dynamic_range_limit", max_batch)
    ceiling = max_batch * max(total_instances, 1) * dynamic_ranges
    return ceiling
