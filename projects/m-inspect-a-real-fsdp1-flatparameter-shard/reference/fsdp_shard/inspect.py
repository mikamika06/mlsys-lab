import math

def inspect_shard(param_sizes, world_size, rank):
    """Inspect FSDP1 flat parameter shard layout."""
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
