def argmin_index(values):
    if not values:
        return -1
    min_val = values[0]
    min_idx = 0
    for i in range(1, len(values)):
        if values[i] < min_val:
            min_val = values[i]
            min_idx = i
    return min_idx

def validate_max_profile_oom(config, limit):
    from sweep.estimator import estimate_memory
    max_mem = estimate_memory(config, 4.0)
    if max_mem > limit:
        raise RuntimeError("OOM at max profile shape")
    return True
