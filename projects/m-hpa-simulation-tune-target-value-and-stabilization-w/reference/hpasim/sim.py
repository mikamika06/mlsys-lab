import math


def simulate_hpa(metrics_history, target_value, stabilization_window_sec, min_replicas, max_replicas, current_replicas):
    replicas_history = []
    last_scale_time = -float("inf")
    current_reps = current_replicas

    for timestamp, metric_val in metrics_history:
        if target_value <= 0:
            desired = min_replicas
        else:
            desired = math.ceil(current_reps * (metric_val / target_value))
        desired = max(min_replicas, min(max_replicas, desired))

        if desired > current_reps:
            current_reps = desired
            last_scale_time = timestamp
            replicas_history.append((timestamp, current_reps))
        elif desired < current_reps:
            if timestamp - last_scale_time >= stabilization_window_sec:
                current_reps = desired
                last_scale_time = timestamp
                replicas_history.append((timestamp, current_reps))
            else:
                replicas_history.append((timestamp, current_reps))
        else:
            replicas_history.append((timestamp, current_reps))
    return replicas_history
