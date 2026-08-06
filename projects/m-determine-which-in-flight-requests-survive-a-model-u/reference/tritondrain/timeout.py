def derive_minimum_drain_timeout(config):
    """Derive minimum required drain timeout in milliseconds."""
    queue_delay = config.get("queue_delay_ms", 0)
    durations = []

    for req in config.get("requests", []):
        st = req.get("stage")
        rem = req.get("remaining_ms", 0)

        if st == "completed":
            continue

        if st == "queued":
            durations.append(rem + queue_delay)
        else:
            durations.append(rem)

    if not durations:
        return 0
    return max(durations)
