def count_recompilations(trace_logs):
    seen = set()
    count = 0
    for log in trace_logs:
        if log.get("is_recompile", False):
            gid = log.get("guard_id")
            if gid not in seen:
                seen.add(gid)
                count += 1
    return count
