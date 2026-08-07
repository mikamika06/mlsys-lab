import random

CONFIGS = [
    {
        "train_batch_size": 64,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 4,
        "zero_optimization": {"stage": 2, "allreduce_bucket_size": 500000000}
    },
    {
        "train_batch_size": 128,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 2,
        "zero_optimization": {"stage": 3, "allreduce_bucket_size": 200000000}
    },
    {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 8,
        "zero_optimization": {"stage": 1, "allreduce_bucket_size": 100000000}
    }
]

TIMELINES = [
    [
        {"name": "compute", "start": 0, "end": 100},
        {"name": "comm", "start": 100, "end": 150},
        {"name": "compute", "start": 150, "end": 250}
    ],
    [
        {"name": "compute", "start": 0, "end": 100},
        {"name": "comm", "start": 50, "end": 120},
        {"name": "compute", "start": 90, "end": 180}
    ]
]


def validate_config(cfg):
    tbs = cfg.get("train_batch_size")
    mbs = cfg.get("train_micro_batch_size_per_gpu")
    gas = cfg.get("gradient_accumulation_steps")
    if tbs is None or mbs is None or gas is None:
        return False
    if tbs != mbs * gas * 2:  # assuming 2 gpus for reference fixture logic
        return False
    zero = cfg.get("zero_optimization")
    if not zero or "stage" not in zero:
        return False
    return True


def compute_overlap_ratio(events):
    total_time = max(e["end"] for e in events) - min(e["start"] for e in events)
    compute_time = sum(e["end"] - e["start"] for e in events if e["name"] == "compute")
    comm_time = sum(e["end"] - e["start"] for e in events if e["name"] == "comm")

    overlap = 0
    for c in events:
        if c["name"] != "compute":

            continue
        for o in events:
            if o["name"] != "comm":
                continue
            latest_start = max(c["start"], o["start"])
            earliest_end = min(c["end"], o["end"])
            if latest_start < earliest_end:
                overlap += (earliest_end - latest_start)
    if total_time == 0:
        return 0.0
    return float(overlap) / float(total_time)


def optimal_bucket_size(tensor_sizes, memory_ceiling):
    best_size = 0
    for size in sorted(tensor_sizes):
        if size <= memory_ceiling:
            best_size = size
    if best_size == 0 and tensor_sizes:
        best_size = min(tensor_sizes)
    return best_size
