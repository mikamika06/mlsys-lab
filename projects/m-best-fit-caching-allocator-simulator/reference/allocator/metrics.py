from allocator.simulator import CachingAllocator


def run_trace(events, segment_size=2097152):
    allocator = CachingAllocator(segment_size=segment_size)
    active_handles = {}

    for event in events:
        op = event[0]
        if op == "alloc":
            tag = event[1]
            size = event[2]
            h = allocator.malloc(size)
            active_handles[tag] = h
        elif op == "free":
            tag = event[1]
            h = active_handles.pop(tag)
            allocator.free(h)

    return {
        "peak_allocated": allocator.peak_allocated,
        "peak_reserved": allocator.peak_reserved,
        "peak_fragmentation": allocator.peak_fragmentation,
        "reserved_deltas": list(allocator.reserved_deltas),
    }
