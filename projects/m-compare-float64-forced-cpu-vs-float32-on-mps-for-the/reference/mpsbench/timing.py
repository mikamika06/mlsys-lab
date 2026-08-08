def analyze_benchmark_trace(trace):
    unsync = float(trace["unsync_measured_ms"])
    sync = float(trace["sync_measured_ms"])
    error_ratio = (sync - unsync) / sync if sync > 0 else 0.0
    illusion_factor = sync / unsync if unsync > 0 else 1.0
    is_valid_measurement = abs(unsync - sync) / sync < 0.05
    return {
        "true_wall_ms": sync,
        "reported_unsync_ms": unsync,
        "underreport_ratio": error_ratio,
        "illusion_factor": illusion_factor,
        "is_valid_measurement": is_valid_measurement,
    }
