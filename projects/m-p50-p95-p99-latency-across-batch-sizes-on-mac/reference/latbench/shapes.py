import numpy as np


def compare_shape_throughput(static_latencies, dynamic_latencies, batch_size):
    """Compare throughput and overhead between static and dynamic tensor allocations."""
    s_arr = np.asarray(static_latencies, dtype=np.float64)
    d_arr = np.asarray(dynamic_latencies, dtype=np.float64)

    s_mean_sec = float(np.mean(s_arr)) / 1000.0
    d_mean_sec = float(np.mean(d_arr)) / 1000.0

    s_tp = float(batch_size) / s_mean_sec if s_mean_sec > 0 else 0.0
    d_tp = float(batch_size) / d_mean_sec if d_mean_sec > 0 else 0.0

    ratio = d_tp / s_tp if s_tp > 0 else 0.0
    overhead = float(np.mean(d_arr) - np.mean(s_arr))

    return {
        "static_throughput": s_tp,
        "dynamic_throughput": d_tp,
        "throughput_ratio": ratio,
        "dynamic_overhead_ms": overhead
    }
