import math
import numpy as np


def calculate_sharded_elements(layer_params, world_size):
    fsdp = sum(math.ceil(p / world_size) for p in layer_params)
    zero3 = math.ceil(sum(layer_params) / world_size)
    return fsdp, zero3


def estimate_zero3_memory(layer_params, world_size):
    _, zero3_per_gpu = calculate_sharded_elements(layer_params, world_size)
    return {
        "params": zero3_per_gpu * 2,
        "grads": zero3_per_gpu * 2,
        "os": zero3_per_gpu * 16,
        "total": zero3_per_gpu * 20
    }


def calculate_prefetch_depth(layer_params, compute_times, bandwidth_bytes_per_sec):
    layer_params = np.array(layer_params, dtype=np.float64)
    compute_times = np.array(compute_times, dtype=np.float64)
    gather_times = (layer_params * 2.0) / bandwidth_bytes_per_sec
    depths = np.zeros(len(layer_params), dtype=np.int32)

    for i in range(1, len(layer_params)):
        d = 1
        accumulated_compute = 0.0
        while d <= i:
            accumulated_compute += compute_times[i - d]
            if accumulated_compute >= gather_times[i]:
                break
            d += 1
        if d > i:
            d = i
        depths[i] = d

    return gather_times.tolist(), depths.tolist()


CONFIGS = [
    {
        "layer_params": [1024, 2048, 4096, 8192, 1024],
        "world_size": 8,
        "compute_times": [0.01, 0.02, 0.04, 0.08, 0.01],
        "bandwidth": 1e9
    },
    {
        "layer_params": [1025, 2049, 4097],
        "world_size": 16,
        "compute_times": [0.05, 0.05, 0.05],
        "bandwidth": 5e8
    },
    {
        "layer_params": [50000] * 10,
        "world_size": 4,
        "compute_times": [0.001] * 10,
        "bandwidth": 1e5
    }
]
