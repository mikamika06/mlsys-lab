import random

CONFIGS = [
    {
        "kernel_name": "vector_add",
        "flops": 1024,
        "bytes_transferred": 4096,
        "peak_flops": 100.0,
        "peak_bandwidth": 10.0,
    },
    {
        "kernel_name": "gemm_kernel",
        "flops": 1048576,
        "bytes_transferred": 32768,
        "peak_flops": 100.0,
        "peak_bandwidth": 10.0,
    },
    {
        "kernel_name": "layer_norm",
        "flops": 4096,
        "bytes_transferred": 8192,
        "peak_flops": 50.0,
        "peak_bandwidth": 5.0,
    },
]

def compute_intensity(flops, bytes_transferred):
    return float(flops) / float(bytes_transferred)

def classify_kernel(flops, bytes_transferred, peak_flops, peak_bandwidth):
    intensity = compute_intensity(flops, bytes_transferred)
    crossover = peak_flops / peak_bandwidth
    if intensity < crossover:
        return "memory-bound", intensity, crossover
    return "compute-bound", intensity, crossover
