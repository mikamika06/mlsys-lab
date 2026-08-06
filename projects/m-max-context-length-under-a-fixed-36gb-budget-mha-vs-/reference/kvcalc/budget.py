from kvcalc.memory import compute_kv_bytes


def max_context_length(config, budget_bytes, batch_size=1):
    low = 1
    high = 10_000_000
    best = 0
    while low <= high:
        mid = (low + high) // 2
        needed = compute_kv_bytes(config, mid, batch_size)
        if needed <= budget_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best
