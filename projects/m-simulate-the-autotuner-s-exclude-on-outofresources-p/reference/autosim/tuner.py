from autosim.config import OutOfResources, evaluate_config


def prune_invalid_configs(configs, workload_size):
    valid = []
    for cfg in configs:
        try:
            evaluate_config(cfg, workload_size)
            valid.append(cfg)
        except OutOfResources:
            pass
    return valid


def autotune(configs, workload_size):
    best_time = float("inf")
    best_idx = -1
    for i, cfg in enumerate(configs):
        try:
            t = evaluate_config(cfg, workload_size)
            if t < best_time:
                best_time = t
                best_idx = i
        except OutOfResources:
            continue
    return {"argmin_index": best_idx, "best_time": best_time if best_idx != -1 else None}
