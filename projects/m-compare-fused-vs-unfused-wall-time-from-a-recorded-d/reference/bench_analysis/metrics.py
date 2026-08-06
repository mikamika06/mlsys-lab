def compute_latency_stats(unfused_ms, fused_ms):
    if fused_ms <= 0:
        speedup = 0.0
    else:
        speedup = unfused_ms / fused_ms
    time_saved = unfused_ms - fused_ms
    return {
        "speedup": float(speedup),
        "time_saved_ms": float(time_saved),
    }


def compute_throughput_gbs(ms_latency, total_bytes):
    if ms_latency <= 0:
        return 0.0
    seconds = ms_latency / 1000.0
    gigabytes = total_bytes / 1e9
    return float(gigabytes / seconds)
