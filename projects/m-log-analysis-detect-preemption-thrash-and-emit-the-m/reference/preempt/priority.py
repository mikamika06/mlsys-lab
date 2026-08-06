def schedule_with_aging(requests, max_blocks, aging_rate):
    """Schedules requests with priority aging and KV block constraints."""
    pending = [dict(r) for r in requests]
    for r in pending:
        r["rem_duration"] = r["duration"]
        r["start_time"] = None
    completion_times = {}
    t = 0
    while any(r["rem_duration"] > 0 for r in pending):
        ready = [r for r in pending if r["arrival_time"] <= t and r["rem_duration"] > 0]
        if not ready:
            t += 1
            continue
        for r in ready:
            r["eff_prio"] = r["priority"] + (t - r["arrival_time"]) * aging_rate
        ready.sort(key=lambda r: (r["eff_prio"], -r["arrival_time"]), reverse=True)

        used_blocks = 0
        for r in ready:
            if used_blocks + r["kv_blocks"] <= max_blocks:
                used_blocks += r["kv_blocks"]
                if r["start_time"] is None:
                    r["start_time"] = t
                r["rem_duration"] -= 1
                if r["rem_duration"] == 0:
                    completion_times[r["id"]] = t + 1
        t += 1

    max_wait = 0
    for r in pending:
        wait = completion_times[r["id"]] - r["arrival_time"] - r["duration"]
        if wait > max_wait:
            max_wait = wait

    return {"completion_times": completion_times, "max_wait_time": max_wait}
