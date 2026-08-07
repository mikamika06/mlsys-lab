def triage_system_state(metrics_snapshot: dict[str, float], baseline_ttft: float) -> str:
    """Diagnose system health state from a snapshot of system metrics."""
    if baseline_ttft <= 0.0:
        return "unknown"
    ttft = float(metrics_snapshot.get("ttft_p99", 0.0))
    kv_usage = float(metrics_snapshot.get("kv_cache_usage", 0.0))
    gpu_util = float(metrics_snapshot.get("gpu_utilization", 0.0))
    prefix_hit = float(metrics_snapshot.get("prefix_hit_rate", 1.0))

    if ttft >= 5.0 * baseline_ttft:
        if kv_usage >= 0.99:
            return "kv_cache_saturated"
        if gpu_util >= 0.95:
            return "gpu_compute_saturated"
        if prefix_hit <= 0.20:
            return "prefix_cache_miss_spike"
        return "unclassified_latency_spike"

    return "nominal"
