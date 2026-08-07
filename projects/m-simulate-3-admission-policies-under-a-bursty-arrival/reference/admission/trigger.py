def identify_trigger_point(log_entries):
    for i, entry in enumerate(log_entries):
        if entry.get("timeout_cascade", False) or entry.get("queue_depth", 0) > 1000:
            return i
    return -1
