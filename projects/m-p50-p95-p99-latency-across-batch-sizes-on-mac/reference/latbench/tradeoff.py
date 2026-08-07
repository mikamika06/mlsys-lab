import numpy as np


def derive_optimal_batch_sizes(profile_summary, max_p99_slo_ms):
    """Derive latency-optimal and throughput-optimal batch sizes under a p99 SLO."""
    valid_batches = [
        b for b, stats in profile_summary.items()
        if stats["p99"] <= max_p99_slo_ms
    ]
    if not valid_batches:
        return {"latency_optimal_b": None, "throughput_optimal_b": None}

    lat_opt = min(valid_batches, key=lambda b: profile_summary[b]["p50"])

    def calc_throughput(b):
        p50_sec = profile_summary[b]["p50"] / 1000.0
        return float(b) / p50_sec if p50_sec > 0 else 0.0

    tp_opt = max(valid_batches, key=calc_throughput)

    return {
        "latency_optimal_b": int(lat_opt),
        "throughput_optimal_b": int(tp_opt)
    }
