def simulate_allocator(trace, total_blocks):
    free_set = set(range(total_blocks))
    allocated = set()
    for op, count in trace:
        if op == "alloc":
            for _ in range(count):
                if not free_set:
                    raise RuntimeError("Out of blocks")
                b = min(free_set)
                free_set.remove(b)
                allocated.add(b)
        elif op == "free":
            for _ in range(count):
                if not allocated:
                    raise RuntimeError("Nothing to free")
                b = min(allocated)
                allocated.remove(b)
                free_set.add(b)
    return len(allocated)
