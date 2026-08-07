def find_largest_context(config, budget_bytes):
    low = 1
    high = 131072
    best = 0
    while low <= high:
        mid = (low + high) // 2
        cost = config["base_bytes"] + mid * config["bytes_per_token"]
        if cost <= budget_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best
