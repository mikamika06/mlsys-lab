import re
import torch

TEST_CASES = [
    {"num_params": 1_000_000, "world_size": 4, "precision_bytes": 2},
    {"num_params": 8_000_000, "world_size": 8, "precision_bytes": 2},
    {"num_params": 500_000, "world_size": 2, "precision_bytes": 4},
]

LOG_SAMPLES = [
    (
        "[INFO] DeepSpeed ZeRO Stage: 1\n[INFO] optimizer_partition: True\n[INFO] estimated memory reduction: 4.0",
        {"zero_stage": 1, "optimizer_partitioned": True, "estimated_reduction": 4.0},
    ),
    (
        "[INFO] ZeRO stage: 0\n[INFO] optimizer_partition: False\n[INFO] estimated memory reduction: 1.0",
        {"zero_stage": 0, "optimizer_partitioned": False, "estimated_reduction": 1.0},
    ),
]


def calculate_zero1_memory(num_params, world_size, precision_bytes=2):
    fp32_bytes = 4
    param_bytes = num_params * precision_bytes
    grad_bytes = num_params * precision_bytes
    opt_state_total_bytes = num_params * (fp32_bytes + fp32_bytes + fp32_bytes)

    baseline_bytes = param_bytes + grad_bytes + opt_state_total_bytes
    opt_state_per_rank = opt_state_total_bytes / world_size
    zero1_bytes = param_bytes + grad_bytes + opt_state_per_rank

    return {
        "baseline_bytes": float(baseline_bytes),
        "zero1_bytes": float(zero1_bytes),
        "opt_state_per_rank_bytes": float(opt_state_per_rank),
    }


def parse_deepspeed_log(log_text):
    stage_match = re.search(r"ZeRO\s+stage\s*:\s*(\d+)", log_text, re.IGNORECASE)
    stage = int(stage_match.group(1)) if stage_match else None

    partition_match = re.search(
        r"optimizer_partition\s*:\s*(True|False)", log_text, re.IGNORECASE
    )
    partitioned = (
        partition_match.group(1).lower() == "true" if partition_match else False
    )

    reduction_match = re.search(
        r"estimated\s+memory\s+reduction\s*:\s*([\d\.]+)", log_text, re.IGNORECASE
    )
    reduction = float(reduction_match.group(1)) if reduction_match else None

    return {
        "zero_stage": stage,
        "optimizer_partitioned": partitioned,
        "estimated_reduction": reduction,
    }
