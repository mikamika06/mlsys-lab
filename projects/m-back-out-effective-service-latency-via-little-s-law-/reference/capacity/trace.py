def compute_effective_latency(trace_data):
    """
    Computes effective service latency W using Little's Law (W = L / lambda).
    trace_data: list of dicts with 'timestamp_sec', 'active_requests', 'completed_requests'
    """
    total_l = 0.0
    total_time = 0.0
    total_completed = 0
    
    for i in range(len(trace_data) - 1):
        dt = trace_data[i + 1]["timestamp_sec"] - trace_data[i]["timestamp_sec"]
        if dt <= 0:
            continue
        total_time += dt
        total_l += trace_data[i]["active_requests"] * dt
        total_completed += trace_data[i]["completed_requests"]
        
    if total_time <= 0 or total_completed <= 0:
        return 0.0
        
    avg_l = total_l / total_time
    arrival_rate = total_completed / total_time
    effective_w = avg_l / arrival_rate
    return effective_w
