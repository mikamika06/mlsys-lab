import numpy as np

GPUS = [
    {"name": "GPU_A", "peak_flop": 100.0, "peak_bw": 10.0, "ridge": 10.0},
    {"name": "GPU_B", "peak_flop": 50.0, "peak_bw": 20.0, "ridge": 2.5}
]

KERNEL = {"flops": 500.0, "bytes_read": 100.0, "bytes_written": 50.0}

PAIRS = [
    {"device": "GPU_A", "flops": 200.0, "bytes": 50.0, "peak_flop": 100.0, "peak_bw": 10.0},
    {"device": "GPU_A", "flops": 1500.0, "bytes": 100.0, "peak_flop": 100.0, "peak_bw": 10.0},
    {"device": "GPU_B", "flops": 100.0, "bytes": 80.0, "peak_flop": 50.0, "peak_bw": 20.0},
    {"device": "GPU_B", "flops": 800.0, "bytes": 50.0, "peak_flop": 50.0, "peak_bw": 20.0},
    {"device": "GPU_A", "flops": 500.0, "bytes": 50.0, "peak_flop": 100.0, "peak_bw": 10.0},
]


def find_flipping_gpu(gpus, kernel):
    total_bytes = kernel["bytes_read"] + kernel["bytes_written"]
    flops = kernel["flops"]
    intensity = flops / total_bytes
    flipping = []
    for gpu in gpus:
        ridge = gpu["peak_flop"] / gpu["peak_bw"]
        is_compute = intensity >= ridge
        if is_compute:
            flipping.append(gpu["name"])
    return flipping[0] if flipping else None


def compute_intensity(bytes_read, bytes_written, flops):
    total_bytes = bytes_read + bytes_written
    if total_bytes == 0:
        return 0.0
    return flops / total_bytes


def classify_bound(intensity, ridge):
    return "compute" if intensity >= ridge else "memory"


def rank_pairs(pairs):
    distances = []
    for p in pairs:
        ridge = p["peak_flop"] / p["peak_bw"]
        intensity = p["flops"] / p["bytes"]
        dist = intensity - ridge
        distances.append((dist, p))
    distances.sort(key=lambda x: x[0])
    return [p for _, p in distances]
