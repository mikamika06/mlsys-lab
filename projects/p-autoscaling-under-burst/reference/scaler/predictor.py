def predict_scaling_action(current_load, trend, threshold, warmup_time_sec):
    projected_load = current_load + trend * warmup_time_sec
    if projected_load > threshold:
        return {"action": "scale_up", "replicas_needed": int(projected_load / threshold) + 1}
    return {"action": "hold", "replicas_needed": 1}
