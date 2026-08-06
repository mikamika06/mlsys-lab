COMPUTE_UNITS = ("cpu_only", "cpu_and_gpu", "all", "cpu_and_ne")


def compute_units_latency(graph_profile, compute_units):
    """Calculate aggregate execution latency for a given compute unit configuration."""
    total_time = 0.0
    current_unit = None
    transfer_cost = graph_profile.get("transfer_cost", 1.5)

    for op in graph_profile.get("nodes", []):
        supported = op.get("supported_units", ["cpu_only"])
        op_time = op.get("op_latency", {})

        if compute_units == "all":
            if "ane" in supported:
                assigned = "ane"
            elif "gpu" in supported:
                assigned = "gpu"
            else:
                assigned = "cpu"
        elif compute_units == "cpu_and_ne":
            if "ane" in supported:
                assigned = "ane"
            else:
                assigned = "cpu"
        elif compute_units == "cpu_and_gpu":
            if "gpu" in supported:
                assigned = "gpu"
            else:
                assigned = "cpu"
        else:
            assigned = "cpu"

        if current_unit is not None and current_unit != assigned:
            total_time += transfer_cost

        current_unit = assigned
        total_time += op_time.get(assigned, op_time.get("cpu", 1.0))

    return total_time


def evaluate_all_units(graph_profile):
    """Evaluate pipeline latency across all four MLComputeUnits settings."""
    return {
        unit: compute_units_latency(graph_profile, unit)
        for unit in COMPUTE_UNITS
    }
