def plan_engine(config, profile, workspace_limit):
    """
    Returns (total_device_memory, total_latency) for the fastest configuration
    that fits the workspace_limit. Memory should be evaluated at max_s.
    If no tactics fit, returns (float('inf'), float('inf')).
    """
    raise NotImplementedError

def sweep_workspace(config, profile, device_memory, limits):
    """
    Returns the index in `limits` that produces the lowest overall latency
    without exceeding `device_memory`. On ties, returns the first index.
    If no limit fits, returns -1.
    """
    raise NotImplementedError
