import numpy as np


def generate_synthetic_data(seed=42):
    rng = np.random.RandomState(seed)
    batch_sizes = [1, 2, 4, 8, 16]
    profile_data = {}
    for b in batch_sizes:
        base_lat = 5.0 + 2.5 * b
        lats = rng.normal(loc=base_lat, scale=0.1 * base_lat, size=200)
        lats = np.maximum(lats, 1.0)
        tail_spikes = rng.exponential(scale=2.0 * b, size=200)
        mask = rng.rand(200) < 0.05
        lats[mask] += tail_spikes[mask]
        profile_data[b] = lats.tolist()

    static_lats = rng.normal(loc=15.0, scale=0.5, size=100).tolist()
    dynamic_lats = [x + float(rng.uniform(1.5, 3.5)) for x in static_lats]

    return {
        "profile_data": profile_data,
        "static_lats": static_lats,
        "dynamic_lats": dynamic_lats,
        "slo": 35.0,
        "shape_batch": 4
    }


def compute_percentiles(latencies_ms):
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99))
    }


def analyze_batch_latencies(batch_profile_data):
    res = {}
    for b, lats in batch_profile_data.items():
        res[int(b)] = compute_percentiles(lats)
    return res


def derive_optimal_batch_sizes(profile_summary, max_p99_slo_ms):
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


def compare_shape_throughput(static_latencies, dynamic_latencies, batch_size):
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
