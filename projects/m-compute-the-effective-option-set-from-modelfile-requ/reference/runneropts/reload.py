def detect_reloads(counters):
    reloads = []
    prev_load = -1
    prev_pid = -1
    for i, c in enumerate(counters):
        curr_load = c.get("load_count", 0)
        curr_pid = c.get("pid", 0)
        if curr_load > prev_load or (prev_pid != -1 and curr_pid != prev_pid):
            if i > 0 or curr_load > 0:
                reloads.append(i)
        prev_load = curr_load
        prev_pid = curr_pid
    return reloads
