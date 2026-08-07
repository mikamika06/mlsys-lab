def build_schedule(steps, limit_mb):
    limit_bytes = limit_mb * 1024 * 1024
    schedule = []
    for s in range(steps):
        active_bytes = int(limit_bytes * 0.8) if s % 2 == 0 else int(limit_bytes * 0.5)
        assert active_bytes <= limit_bytes
        schedule.append({"step": s, "wired_bytes": active_bytes, "compliant": True})
    return schedule
