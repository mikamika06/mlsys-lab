import math

MODELS = [
    {"name": "small", "params": [1024, 2048, 512, 128]},
    {"name": "medium", "params": [4096, 4096, 8192, 1024, 256]},
    {"name": "lopsided", "params": [16384, 512, 512, 512]}
]

def inspect_shard(param_sizes, world_size, rank):
    total = sum(param_sizes)
    padded_total = math.ceil(total / world_size) * world_size
    shard_size = padded_total // world_size
    start = rank * shard_size
    end = start + shard_size
    return {
        "total_params": total,
        "padded_total": padded_total,
        "shard_size": shard_size,
        "start": start,
        "end": end
    }

def compute_padding_overhead(param_sizes, world_size):
    total = sum(param_sizes)
    padded_total = math.ceil(total / world_size) * world_size
    padding = padded_total - total
    return float(padding) / float(padded_total) if padded_total > 0 else 0.0

def per_rank_balance(param_sizes, world_size):
    total = sum(param_sizes)
    padded_total = math.ceil(total / world_size) * world_size
    shard_size = padded_total // world_size
    return [shard_size for _ in range(world_size)]
