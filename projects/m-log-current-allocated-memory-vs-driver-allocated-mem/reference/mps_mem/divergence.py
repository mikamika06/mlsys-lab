def log_divergence(device, ops):
    log = []
    tensors = {}
    for op in ops:
        if op[0] == "alloc":
            tensors[op[2]] = device.allocate(op[1])
        elif op[0] == "free":
            device.free(tensors.pop(op[1]))
        log.append((device.current_allocated_memory(), device.driver_allocated_memory()))
    return log
