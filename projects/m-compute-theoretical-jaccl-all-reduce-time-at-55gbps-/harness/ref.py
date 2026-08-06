import math

TEST_CASES_M1 = [
    {"tensor_bytes": 1024 * 1024 * 128, "num_ranks": 2, "bandwidth_gbps": 55.0},
    {"tensor_bytes": 1024 * 1024 * 512, "num_ranks": 4, "bandwidth_gbps": 55.0},
    {"tensor_bytes": 1024 * 1024 * 1024, "num_ranks": 8, "bandwidth_gbps": 80.0},
]

TEST_CASES_M2 = [
    {
        "tensor_bytes": 1024 * 1024 * 256,
        "num_ranks": 2,
        "latency_per_step_sec": 5e-6,
        "compute_time_sec": 0.015,
        "bandwidth_gbps": 55.0,
    },
    {
        "tensor_bytes": 1024 * 1024 * 512,
        "num_ranks": 2,
        "latency_per_step_sec": 8e-6,
        "compute_time_sec": 0.025,
        "bandwidth_gbps": 55.0,
    },
    {
        "tensor_bytes": 1024 * 1024 * 1024,
        "num_ranks": 4,
        "latency_per_step_sec": 12e-6,
        "compute_time_sec": 0.040,
        "bandwidth_gbps": 55.0,
    },
]


def ref_allreduce_time(tensor_bytes, num_ranks, bandwidth_gbps=55.0):
    bytes_transferred = 2.0 * (num_ranks - 1) / num_ranks * tensor_bytes
    bw_bytes_sec = (bandwidth_gbps * 1e9) / 8.0
    return bytes_transferred / bw_bytes_sec


def ref_allreduce_overhead(tensor_bytes, num_ranks, latency_per_step_sec, compute_time_sec, bandwidth_gbps=55.0):
    transfer = ref_allreduce_time(tensor_bytes, num_ranks, bandwidth_gbps)
    hop_latency = 2.0 * (num_ranks - 1) * latency_per_step_sec
    return (transfer + hop_latency) / compute_time_sec


def ref_min_microbatches(num_ranks, target_bubble_fraction):
    val = ((num_ranks - 1) * (1.0 - target_bubble_fraction)) / target_bubble_fraction
    return max(1, math.ceil(val - 1e-9))
