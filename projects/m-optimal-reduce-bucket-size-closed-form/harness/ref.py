import numpy as np
import re

LOG_FIXTURES = [
    ("""
    DeepSpeed ZeRO-2 Memory Estimator Output
    Total Number of Parameters: 1_500_000_000
    ZeRO Stage: 2
    World Size: 4
    Base Memory: 3.00 GB
    Gradient Memory: 0.75 GB
    Total Memory: 3.75 GB
    """, {
        "params_numel": 1500000000,
        "zero_stage": 2,
        "world_size": 4,
        "base_mem_gb": 3.0,
        "grad_mem_gb": 0.75,
        "total_mem_gb": 3.75
    }),
    ("""
    DeepSpeed ZeRO-2 Memory Estimator Output
    Total Number of Parameters: 13000000000
    ZeRO Stage: 2
    World Size: 64
    Base Memory: 26.00 GB
    Gradient Memory: 0.41 GB
    Total Memory: 26.41 GB
    """, {
        "params_numel": 13000000000,
        "zero_stage": 2,
        "world_size": 64,
        "base_mem_gb": 26.0,
        "grad_mem_gb": 0.41,
        "total_mem_gb": 26.41
    })
]

PARAM_SHAPE_FIXTURES = [
    [(128, 64), (3, 17), (1024, 512), (1, 1)],
    [(4096, 4096), (4096, 11008), (11008, 4096), (1, 4096)]
]

PLANNER_FIXTURES = [
    {
        "total_params": 1_000_000_000,
        "num_ranks": 8,
        "latency_sec": 5e-6,
        "bandwidth_bytes_per_sec": 25e9,
        "elem_bytes": 2,
        "max_mem_bytes": 500_000_000
    },
    {
        "total_params": 7_000_000_000,
        "num_ranks": 64,
        "latency_sec": 2e-5,
        "bandwidth_bytes_per_sec": 100e9,
        "elem_bytes": 2,
        "max_mem_bytes": 100_000_000
    }
]

def parse_memory_estimator_log(log_text):
    res = {
        "params_numel": 0,
        "zero_stage": 0,
        "world_size": 1,
        "base_mem_gb": 0.0,
        "grad_mem_gb": 0.0,
        "total_mem_gb": 0.0
    }
    p_match = re.search(r"Total\s+Number\s+of\s+Parameters:\s*([\d_]+|[\d]+)", log_text, re.IGNORECASE)
    if p_match:
        res["params_numel"] = int(p_match.group(1).replace("_", ""))
    stage_match = re.search(r"ZeRO\s*Stage:\s*(\d+)", log_text, re.IGNORECASE)
    if stage_match:
        res["zero_stage"] = int(stage_match.group(1))
    ws_match = re.search(r"World\s*Size:\s*(\d+)", log_text, re.IGNORECASE)
    if ws_match:
        res["world_size"] = int(ws_match.group(1))
    base_match = re.search(r"Base\s+Memory:\s*([\d\.]+)\s*GB", log_text, re.IGNORECASE)
    if base_match:
        res["base_mem_gb"] = float(base_match.group(1))
    grad_match = re.search(r"Gradient\s+Memory:\s*([\d\.]+)\s*GB", log_text, re.IGNORECASE)
    if grad_match:
        res["grad_mem_gb"] = float(grad_match.group(1))
    tot_match = re.search(r"Total\s+Memory:\s*([\d\.]+)\s*GB", log_text, re.IGNORECASE)
    if tot_match:
        res["total_mem_gb"] = float(tot_match.group(1))
    return res

def compute_fragmentation_curve(param_shapes, elem_bytes=2, alignment_bytes=512):
    exact_bytes = []
    padded_bytes = []
    cum_exact = 0
    cum_padded = 0
    for shape in param_shapes:
        numel = int(np.prod(shape))
        raw_b = numel * elem_bytes
        pad_b = int(np.ceil(raw_b / alignment_bytes) * alignment_bytes)
        cum_exact += raw_b
        cum_padded += pad_b
        exact_bytes.append(cum_exact)
        padded_bytes.append(cum_padded)
    frag_ratios = [
        (p - e) / float(p) if p > 0 else 0.0
        for e, p in zip(exact_bytes, padded_bytes)
    ]
    return {
        "cumulative_exact_bytes": exact_bytes,
        "cumulative_padded_bytes": padded_bytes,
        "fragmentation_ratio_curve": frag_ratios,
        "total_overhead_bytes": cum_padded - cum_exact
    }

def compute_optimal_reduce_bucket_size(total_params, num_ranks, latency_sec, bandwidth_bytes_per_sec, elem_bytes=2, max_mem_bytes=1073741824):
    total_grad_bytes = total_params * elem_bytes
    if num_ranks <= 1:
        return min(total_params, int(max_mem_bytes // elem_bytes))
    coeff = 2.0 * (num_ranks - 1) / float(num_ranks)
    def comm_time(b_bytes):
        num_buckets = int(np.ceil(total_grad_bytes / b_bytes))
        return num_buckets * (latency_sec + coeff * (b_bytes / bandwidth_bytes_per_sec))
    min_b = 1024
    max_b = min(total_grad_bytes, max_mem_bytes)
    if min_b >= max_b:
        return int(max_b // elem_bytes)
    candidates = np.logspace(np.log10(min_b), np.log10(max_b), num=200)
    best_b = max_b
    best_t = float('inf')
    for b in candidates:
        b_aligned = int(np.floor(b / 512.0) * 512)
        if b_aligned < min_b:
            b_aligned = min_b
        if b_aligned > max_b:
            b_aligned = max_b
        t = comm_time(b_aligned)
        if t < best_t:
            best_t = t
            best_b = b_aligned
    optimal_numel = int(best_b // elem_bytes)
    return optimal_numel
