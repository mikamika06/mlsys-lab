def count_sync_points(trace):
    count = 0
    for item in trace:
        if item.get("type") == "sync":
            count += 1
    return count
