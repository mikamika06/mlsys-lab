import re


def parse_deepspeed_init_log(log_lines):
    partitions = {}
    for line in log_lines:
        match = re.search(r"Rank\s+(\d+):\s+partition_size\s*=\s*(\d+)", line)
        if match:
            rank = int(match.group(1))
            size = int(match.group(2))
            partitions[rank] = size
    return partitions


def parse_zero_runtime_log(log_lines):
    reductions = {}
    for line in log_lines:
        match = re.search(r"Rank\s+(\d+):\s+memory_reduction\s*=\s*([0-9.]+)", line)
        if match:
            rank = int(match.group(1))
            val = float(match.group(2))
            reductions[rank] = val
    return reductions
