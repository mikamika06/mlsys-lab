def optimize_instance_ratio(total_cores, target_instances):
    cores_per_instance = max(1, total_cores // target_instances)
    return {"instances": target_instances, "threads_per_instance": cores_per_instance}


def verify_high_scaling(scaling_data, threshold):
    threads = sorted(list(scaling_data.keys()))
    if len(threads) < 2:
        return True
    t_min, t_max = threads[0], threads[-1]
    tp_min, tp_max = scaling_data[t_min], scaling_data[t_max]
    ratio = float(t_max) / float(t_min)
    tp_ratio = tp_max / tp_min if tp_min > 0 else 0.0
    return (tp_ratio / ratio) >= threshold
