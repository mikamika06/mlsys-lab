import random

CONFIGS = [
    {"pp_size": 4, "microbatches": 16, "total_flops": 2e15, "time_per_stage": 0.05},
    {"pp_size": 8, "microbatches": 32, "total_flops": 5e15, "time_per_stage": 0.08},
    {"pp_size": 2, "microbatches": 8, "total_flops": 1e15, "time_per_stage": 0.02},
]

LOG_SAMPLES = [
    [
        {"stage": 0, "activation_bytes": 1024},
        {"stage": 1, "activation_bytes": 4096},
        {"stage": 2, "activation_bytes": 2048},
        {"stage": 3, "activation_bytes": 1536},
    ],
    [
        {"stage": 0, "activation_bytes": 8192},
        {"stage": 1, "activation_bytes": 2048},
    ]
]


def compute_throughput(pp_size, microbatches, total_flops, time_per_stage):
    bubble_fraction = (pp_size - 1.0) / float(microbatches)
    effective_time = time_per_stage * (1.0 + bubble_fraction)
    return float(total_flops / effective_time / 1e12)


def find_imbalanced_stage(logs):
    stage_allocations = {}
    for entry in logs:
        stage = entry["stage"]
        mem = entry["activation_bytes"]
        stage_allocations.setdefault(stage, []).append(mem)
    means = {s: sum(m) / len(m) for s, m in stage_allocations.items()}
    return int(max(means, key=means.get))


def peak_memory_1f1b(pp_size, microbatches, hidden_size, seq_len, batch_size):
    base = batch_size * seq_len * hidden_size * 4
    return int(base * (pp_size + microbatches))


def peak_memory_interleaved(pp_size, microbatches, virtual_pp_stages, hidden_size, seq_len, batch_size):
    base = batch_size * seq_len * hidden_size * 4
    return int(base * (pp_size + microbatches / float(virtual_pp_stages)))
