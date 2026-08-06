import math

CONFIGS = [
    {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64, "element_bytes": 2, "block_r": 64, "block_c": 64},
    {"batch_size": 4, "num_heads": 16, "seq_len": 2048, "head_dim": 128, "element_bytes": 2, "block_r": 128, "block_c": 128},
    {"batch_size": 2, "num_heads": 32, "seq_len": 4096, "head_dim": 64, "element_bytes": 4, "block_r": 256, "block_c": 128},
    {"batch_size": 8, "num_heads": 12, "seq_len": 1024, "head_dim": 64, "element_bytes": 2, "block_r": 64, "block_c": 64},
]

EXECUTION_CASES = [
    {"flops": 1e12, "bytes_transferred": 5e9, "execution_time_sec": 0.005, "peak_tflops": 312.0, "peak_gbps": 2000.0},
    {"flops": 10e12, "bytes_transferred": 2e9, "execution_time_sec": 0.035, "peak_tflops": 312.0, "peak_gbps": 2000.0},
    {"flops": 0.5e12, "bytes_transferred": 10e9, "execution_time_sec": 0.010, "peak_tflops": 150.0, "peak_gbps": 900.0},
]


def compute_bytes_transferred(config):
    b = config["batch_size"]
    h = config["num_heads"]
    n = config["seq_len"]
    d = config["head_dim"]
    p = config["element_bytes"]
    br = config["block_r"]
    tr = math.ceil(n / br)

    naive = float(b * h * p * (4 * n * d + 2 * n * n))
    tiled = float(b * h * n * d * p * (2 + 2 * tr))
    return {"naive_bytes": naive, "tiled_bytes": tiled}


def compute_achieved_bandwidth(bytes_transferred, execution_time_sec):
    return float((bytes_transferred / 1e9) / execution_time_sec)


def compute_bandwidth_efficiency(achieved_gbps, peak_gbps):
    return float(achieved_gbps / peak_gbps)


def compute_arithmetic_intensity(flops, bytes_transferred):
    return float(flops / bytes_transferred)


def compute_roofline_bound(intensity, peak_tflops, peak_gbps):
    mem_bound_tflops = (intensity * peak_gbps) / 1000.0
    is_mem_bound = mem_bound_tflops < peak_tflops
    attainable_tflops = min(float(peak_tflops), float(mem_bound_tflops))
    knee_intensity = (peak_tflops * 1000.0) / peak_gbps
    return {
        "attainable_tflops": float(attainable_tflops),
        "is_memory_bound": bool(is_mem_bound),
        "knee_intensity": float(knee_intensity),
    }


def analyze_kernel_execution(flops, bytes_transferred, execution_time_sec, peak_tflops, peak_gbps):
    achieved_tflops = (flops / 1e12) / execution_time_sec
    achieved_gbps = (bytes_transferred / 1e9) / execution_time_sec
    intensity = compute_arithmetic_intensity(flops, bytes_transferred)
    bound = compute_roofline_bound(intensity, peak_tflops, peak_gbps)
    pct_peak_bw = achieved_gbps / peak_gbps
    pct_attainable = achieved_tflops / bound["attainable_tflops"]

    return {
        "achieved_tflops": float(achieved_tflops),
        "achieved_gbps": float(achieved_gbps),
        "arithmetic_intensity": float(intensity),
        "is_memory_bound": bool(bound["is_memory_bound"]),
        "knee_intensity": float(bound["knee_intensity"]),
        "attainable_tflops": float(bound["attainable_tflops"]),
        "pct_peak_bandwidth": float(pct_peak_bw),
        "pct_attainable_performance": float(pct_attainable),
    }
