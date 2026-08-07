def plan_engine(config, profile, workspace_limit):
    """
    Returns (total_device_memory, total_latency) for the fastest configuration
    that fits the workspace_limit. Memory should be evaluated at max_s.
    If no tactics fit, returns (float('inf'), float('inf')).
    """
    max_s = profile["max_s"]
    opt_s = profile["opt_s"]
    max_ws = 0
    total_lat = 0.0

    for layer in config["layers"]:
        best_lat = float('inf')
        chosen_ws = 0
        for t in layer["tactics"]:
            ws = t["base_ws"] + t["ws_factor"] * max_s
            if ws <= workspace_limit:
                lat = t["base_lat"] + t["lat_factor"] * opt_s
                if lat < best_lat:
                    best_lat = lat
                    chosen_ws = ws
        if best_lat == float('inf'):
            return float('inf'), float('inf')
        total_lat += best_lat
        max_ws = max(max_ws, chosen_ws)

    return config["weights_memory"] + max_ws, total_lat

def sweep_workspace(config, profile, device_memory, limits):
    """
    Returns the index in `limits` that produces the lowest overall latency
    without exceeding `device_memory`. On ties, returns the first index.
    If no limit fits, returns -1.
    """
    best_idx = -1
    best_lat = float('inf')

    for i, limit in enumerate(limits):
        mem, lat = plan_engine(config, profile, limit)
        if mem <= device_memory and lat < best_lat:
            best_lat = lat
            best_idx = i

    return best_idx
