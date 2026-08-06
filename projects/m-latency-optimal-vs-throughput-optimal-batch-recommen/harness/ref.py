import numpy as np

PROFILES = [
    {
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "draft_accept_rate": 0.7,
        "base_latency_ms": 10.0,
        "overhead_ms": 2.0,
    },
    {
        "batch_sizes": [1, 4, 8, 16, 32, 64],
        "draft_accept_rate": 0.5,
        "base_latency_ms": 15.0,
        "overhead_ms": 3.5,
    },
]


def compute_curves(profile):
    bs = np.array(profile["batch_sizes"], dtype=float)
    accept = profile["draft_accept_rate"]
    base = profile["base_latency_ms"]
    ov = profile["overhead_ms"]
    latencies = base + ov * np.sqrt(bs) / (1.0 + accept)
    throughput = bs / (latencies / 1000.0)
    return latencies, throughput


def compute_metrics(profile):
    lat, tp = compute_curves(profile)
    return {"latencies": lat.tolist(), "throughput": tp.tolist()}


def recommend_batches(profile):
    bs = np.array(profile["batch_sizes"], dtype=int)
    lat, tp = compute_curves(profile)
    lat_opt_idx = int(np.argmin(lat))
    tp_opt_idx = int(np.argmax(tp))
    return {
        "latency_optimal_batch": int(bs[lat_opt_idx]),
        "throughput_optimal_batch": int(bs[tp_opt_idx]),
    }
