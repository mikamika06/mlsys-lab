PEAK_DRAM_BW_GBPS = 1500.0
PEAK_COMPUTE_TFLOPS = 312.0

KERNELS = [
    {"name": "vector_add", "dram_pct": 85.0, "compute_pct": 10.0, "duration_us": 5.0, "bytes": 1024000, "time_ns": 5000, "flops": 0},
    {"name": "matrix_multiply", "dram_pct": 20.0, "compute_pct": 92.0, "duration_us": 50.0, "bytes": 512000, "time_ns": 50000, "flops": 1000000000},
    {"name": "small_reduction", "dram_pct": 5.0, "compute_pct": 2.0, "duration_us": 1.2, "bytes": 4096, "time_ns": 1200, "flops": 1000},
    {"name": "activation_gelu", "dram_pct": 78.0, "compute_pct": 35.0, "duration_us": 8.0, "bytes": 2048000, "time_ns": 8000, "flops": 500000},
    {"name": "layer_norm", "dram_pct": 90.0, "compute_pct": 15.0, "duration_us": 12.0, "bytes": 4096000, "time_ns": 12000, "flops": 200000},
    {"name": "attention_score", "dram_pct": 60.0, "compute_pct": 70.0, "duration_us": 30.0, "bytes": 8192000, "time_ns": 30000, "flops": 250000000}
]


def reference_classify(kernels):
    out = []
    for k in kernels:
        if k["duration_us"] < 2.0 and k["dram_pct"] < 10.0 and k["compute_pct"] < 10.0:
            out.append("latency-bound")
        elif k["dram_pct"] >= k["compute_pct"]:
            out.append("memory-bound")
        else:
            out.append("compute-bound")
    return out


def reference_achieved_gbps(dram_pct, peak_gbps):
    return (dram_pct / 100.0) * peak_gbps


def reference_cross_check(bytes_sum, time_ns):
    return (bytes_sum / 1e9) / (time_ns / 1e9)


def reference_rank(kernels, peak_tflops):
    scored = []
    for k in kernels:
        gfs = (k["compute_pct"] / 100.0) * peak_tflops * 1000.0
        scored.append((k["name"], gfs))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored]
