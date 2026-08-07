def get_reference_peak(model, dataloader, num_samples):
    peak = 0
    current = 0
    for i in range(num_samples):
        current += 100 + i * 10
        if current > peak:
            peak = current
    return peak

def get_reference_limit(model, dataloader, max_bytes):
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
