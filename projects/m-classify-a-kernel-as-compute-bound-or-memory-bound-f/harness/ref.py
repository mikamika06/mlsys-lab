HARDWARE_SPECS = [
    {"name": "gpu_a", "peak_flops": 312e12, "peak_bandwidth": 1.55e12, "ridge_point": 201.29},
    {"name": "gpu_b", "peak_flops": 125e12, "peak_bandwidth": 900e9, "ridge_point": 138.89},
    {"name": "gpu_c", "peak_flops": 65e12, "peak_bandwidth": 440e9, "ridge_point": 147.73},
]

TRACES = [
    {"name": "attention_forward", "flops": 4500000000, "bytes_transferred": 15000000},
    {"name": "layer_norm", "flops": 120000000, "bytes_transferred": 24000000},
    {"name": "gelu_activation", "flops": 50000000, "bytes_transferred": 10000000},
    {"name": "gemm_kernel", "flops": 80000000000, "bytes_transferred": 100000000},
]

def get_oracle_data():
    results = []
    for trace in TRACES:
        spec = HARDWARE_SPECS[0]
        flops = trace["flops"]
        bytes_tx = trace["bytes_transferred"]
        intensity = float(flops) / float(bytes_tx)
        bound = "compute-bound" if intensity >= spec["ridge_point"] else "memory-bound"
        results.append({
            "name": trace["name"],
            "intensity": intensity,
            "classification": bound
        })
    return results
