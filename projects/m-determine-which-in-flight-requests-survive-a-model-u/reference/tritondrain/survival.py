def determine_surviving_requests(config):
    """Determine list of request IDs that survive a model unload operation."""
    mode = config.get("unload_mode", "EXPLICIT")
    grace = config.get("grace_period_ms", 0) if mode == "GRACEFUL" else 0
    survivors = []

    for req in config.get("requests", []):
        st = req.get("stage")
        rem = req.get("remaining_ms", 0)

        if st == "completed":
            survivors.append(req["id"])
        elif mode == "EXPLICIT":
            if st == "executing" and rem == 0:
                survivors.append(req["id"])
        elif mode == "GRACEFUL":
            if rem <= grace:
                survivors.append(req["id"])

    return sorted(survivors)
