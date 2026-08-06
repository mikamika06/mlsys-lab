def find_cpu_fallback_op(graph_profile):
    """Identify the first graph operation forcing a CPU fallback segment."""
    current_unit = None
    for op in graph_profile.get("nodes", []):
        supported = op.get("supported_units", ["cpu_only"])
        if "ane" in supported:
            assigned = "ane"
        elif "gpu" in supported:
            assigned = "gpu"
        else:
            assigned = "cpu"

        if (
            current_unit is not None
            and current_unit != "cpu"
            and assigned == "cpu"
        ):
            return op["name"]
        current_unit = assigned
    return None
