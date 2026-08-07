def parse_concurrency_ceiling(config: dict) -> dict:
    groups = config.get("instance_group", [])
    if not groups:
        total_instances = 1
    else:
        total_instances = 0
        for g in groups:
            count = g.get("count", 1)
            gpus = g.get("gpus")
            gpu_count = len(gpus) if gpus else 1
            total_instances += count * gpu_count

    mb = config.get("max_batch_size", 0)
    eff_batch = max(1, mb)
    has_db = "dynamic_batching" in config
    db_cfg = config.get("dynamic_batching") or {}
    max_queue_delay = db_cfg.get("max_queue_delay_microseconds", 0) if has_db else 0

    return {
        "total_instances": total_instances,
        "max_batch_size": mb,
        "effective_batch_size": eff_batch,
        "concurrency_ceiling": total_instances * eff_batch,
        "has_dynamic_batching": has_db,
        "max_queue_delay_us": max_queue_delay,
    }
