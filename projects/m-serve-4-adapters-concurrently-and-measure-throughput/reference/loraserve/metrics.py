def compute_throughput_ratio(multi_metrics, base_metrics):
    multi_tp = multi_metrics.get("throughput", 0.0)
    base_tp = base_metrics.get("throughput", 1.0)
    if base_tp == 0:
        return 0.0
    return multi_tp / base_tp
