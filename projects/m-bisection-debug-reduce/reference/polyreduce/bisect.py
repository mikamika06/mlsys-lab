def bisect_divergent_step(num_steps, check_step_fn):
    """Find the first divergent step index using binary search."""
    if num_steps <= 0:
        return -1
    low = 0
    high = num_steps - 1
    result = -1
    while low <= high:
        mid = (low + high) // 2
        if not check_step_fn(mid):
            result = mid
            high = mid - 1
        else:
            low = mid + 1
    return result
