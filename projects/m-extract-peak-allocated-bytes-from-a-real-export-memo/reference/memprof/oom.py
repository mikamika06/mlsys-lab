def find_largest_live_allocation(oom_snapshot):
    allocations = oom_snapshot.get("allocations", [])
    largest = None
    max_size = -1
    for alloc in allocations:
        if alloc.get("status") == "live":
            size = alloc.get("size", 0)
            if size > max_size:
                max_size = size
                largest = alloc
    return largest
