def forecast_scaling(world_size_target, baseline_metrics):
    base_size = baseline_metrics.get("world_size", 8)
    base_time = baseline_metrics.get("step_time", 1.0)
    comm_fraction = baseline_metrics.get("comm_fraction", 0.2)
    comp_fraction = 1.0 - comm_fraction
    predicted_comm = comm_fraction * base_time * (world_size_target / base_size)
    predicted_step = (comp_fraction * base_time) + predicted_comm
    return {
        "world_size": world_size_target,
        "predicted_step_time": predicted_step,
        "speedup": (base_time * base_size) / (predicted_step * world_size_target)
    }
