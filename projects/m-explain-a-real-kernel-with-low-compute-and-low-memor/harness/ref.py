KERNELS = [
    {
        "id": "k1",
        "flops": 1.2e8,
        "bytes_transferred": 4.0e6,
        "duration_ms": 0.05,
        "peak_gflops": 312.0,
        "peak_bandwidth_gbps": 900.0,
        "active_warps": 4,
        "max_warps": 32,
    },
    {
        "id": "k2",
        "flops": 5.0e7,
        "bytes_transferred": 2.0e6,
        "duration_ms": 0.02,
        "peak_gflops": 312.0,
        "peak_bandwidth_gbps": 900.0,
        "active_warps": 2,
        "max_warps": 32,
    },
    {
        "id": "k3",
        "flops": 8.0e7,
        "bytes_transferred": 1.5e6,
        "duration_ms": 0.03,
        "peak_gflops": 312.0,
        "peak_bandwidth_gbps": 900.0,
        "active_warps": 8,
        "max_warps": 32,
    },
]


def compute_metrics(k):
    gflops = (k["flops"] / (k["duration_ms"] * 1e-3)) / 1e9
    intensity = k["flops"] / k["bytes_transferred"]
    achieved_bw = (k["bytes_transferred"] / (k["duration_ms"] * 1e-3)) / 1e9
    compute_pct = (gflops / k["peak_gflops"]) * 100.0
    memory_pct = (achieved_bw / k["peak_bandwidth_gbps"]) * 100.0
    gap_pct = 100.0 - compute_pct
    return {
        "gflops": gflops,
        "intensity": intensity,
        "compute_pct": compute_pct,
        "memory_pct": memory_pct,
        "gap_pct": gap_pct,
    }


def classify_bottleneck(k):
    m = compute_metrics(k)
    occupancy_ratio = k["active_warps"] / k["max_warps"]
    if m["compute_pct"] < 25.0 and m["memory_pct"] < 25.0 and occupancy_ratio < 0.3:
        return "latency-bound-low-occupancy"
    elif m["compute_pct"] >= m["memory_pct"]:
        return "compute-bound"
    else:
        return "memory-bound"


def generate_report(k):
    m = compute_metrics(k)
    b = classify_bottleneck(k)
    return {
        "id": k["id"],
        "compute_pct": round(m["compute_pct"], 2),
        "memory_pct": round(m["memory_pct"], 2),
        "bottleneck": b,
    }
