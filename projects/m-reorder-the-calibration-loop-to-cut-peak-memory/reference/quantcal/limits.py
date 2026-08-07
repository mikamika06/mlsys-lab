def max_workable_samples(model, dataloader, max_bytes):
    low = 0
    high = 100
    best = 0
    while low <= high:
        mid = (low + high) // 2
        cost = 50 * mid
        if cost <= max_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best
