WORKLOADS = [
    [("alloc", 1000, "a"), ("alloc", 2000, "b"), ("free", "a"), ("alloc", 1500, "c")],
    [("alloc", 5000, "t1"), ("free", "t1"), ("alloc", 5000, "t2"), ("free", "t2")],
    [("alloc", 100, "x"), ("alloc", 100, "y"), ("alloc", 100, "z"), ("free", "x"), ("free", "z"), ("alloc", 250, "w")]
]


def expected_divergence(device_cls, ops):
    device = device_cls()
    log = []
    tensors = {}
    for op in ops:
        if op[0] == "alloc":
            tensors[op[2]] = device.allocate(op[1])
        elif op[0] == "free":
            device.free(tensors.pop(op[1]))
        log.append((device.current_allocated_memory(), device.driver_allocated_memory()))
    return log
