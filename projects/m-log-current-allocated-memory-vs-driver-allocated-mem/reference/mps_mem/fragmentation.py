def reproduce_oom(device):
    chunk = device.recommended_max_memory() // 3
    t1 = device.allocate(chunk)
    t2 = device.allocate(chunk)
    t3 = device.allocate(chunk)
    device.free(t1)
    device.free(t3)
    device.allocate(chunk * 2)


def fix_oom(device):
    chunk = device.recommended_max_memory() // 3
    t1 = device.allocate(chunk)
    t2 = device.allocate(chunk)
    t3 = device.allocate(chunk)
    device.free(t1)
    device.free(t3)
    device.empty_cache()
    device.allocate(chunk * 2)
