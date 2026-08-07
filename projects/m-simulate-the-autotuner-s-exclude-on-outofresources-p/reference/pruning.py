class OutOfResources(Exception):
    pass


def is_dominated(config, oom_configs, resource_keys):
    """
    Returns True if `config` is dominated by ANY config in `oom_configs`.
    A config is dominated if, for ALL keys in `resource_keys`,
    config[key] >= oom_config[key].
    """
    for oom in oom_configs:
        if all(config[k] >= oom[k] for k in resource_keys):
            return True
    return False


def autotune(configs, evaluate, resource_keys):
    """
    Evaluate configs to find the one with the minimum execution time.
    If `evaluate(config)` raises OutOfResources, record it.
    Skip any config that is dominated by a known OOM config.

    Returns the original index (in `configs`) of the best config.
    If no configs succeed, return -1.
    """
    oom_configs = []
    best_idx = -1
    best_time = float('inf')

    for i, config in enumerate(configs):
        if is_dominated(config, oom_configs, resource_keys):
            continue
        try:
            time = evaluate(config)
            if time < best_time:
                best_time = time
                best_idx = i
        except OutOfResources:
            oom_configs.append(config)

    return best_idx
