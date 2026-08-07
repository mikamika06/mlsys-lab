def compute_overhead(profile):
    total_nodes = sum(n["dur"] for n in profile["nodes"])
    return profile["session_duration"] - total_nodes
