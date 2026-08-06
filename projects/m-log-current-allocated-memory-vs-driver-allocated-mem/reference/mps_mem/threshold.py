from mps_mem.mock_device import OutOfMemoryError


def find_oom_threshold(device):
    low = 0
    high = 100_000_000
    best = 0
    while low <= high:
        mid = (low + high) // 2
        try:
            tid = device.allocate(mid)
            best = mid
            device.free(tid)
            device.empty_cache()
            low = mid + 1
        except OutOfMemoryError:
            high = mid - 1
            device.empty_cache()
    return best, device.recommended_max_memory()
