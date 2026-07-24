def fragmentation_gap(trace):
    current = 0
    max_allocated = 0
    reserved = 0
    alloc_sizes = {}
    next_id = 0
    for op, val in trace:
        if op == "alloc":
            size = val
            reserved += size
            current += size
            alloc_sizes[next_id] = size
            if current > max_allocated:
                max_allocated = current
            next_id += 1
        elif op == "free":
            id_ = val
            size = alloc_sizes[id_]
            current -= size
    return reserved - max_allocated
