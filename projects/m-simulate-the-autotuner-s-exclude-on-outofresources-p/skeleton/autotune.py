class OutOfResources(Exception):
    pass


def is_dominated(config, oom_configs, resource_keys):
    """
    Returns True if `config` is dominated by ANY config in `oom_configs`.
    A config is dominated if, for ALL keys in `resource_keys`,
    config[key] >= oom_config[key].
    """
    raise NotImplementedError


def autotune(configs, evaluate, resource_keys):
    """
    Evaluate configs to find the one with the minimum execution time.
    If `evaluate(config)` raises OutOfResources, remember it.
    Skip any config that is dominated by a known OOM config.

    Returns the original index (in `configs`) of the best config.
    If no configs succeed, return -1.
    """
    raise NotImplementedError
