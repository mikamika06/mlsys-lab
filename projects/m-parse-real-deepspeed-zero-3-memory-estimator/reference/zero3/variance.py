import numpy as np


def zero3_vs_fsdp2_variance(tensor_shapes, world_size, bytes_per_elem=2):
    zero3_rank_sizes = np.zeros(world_size, dtype=np.int64)
    for shape in tensor_shapes:
        numel = int(np.prod(shape))
        shard_size = int(np.ceil(numel / world_size))
        for r in range(world_size):
            zero3_rank_sizes[r] += shard_size

    total_numel = sum(int(np.prod(s)) for s in tensor_shapes)
    fsdp2_rank_sizes = np.zeros(world_size, dtype=np.int64)
    base_shard = total_numel // world_size
    rem = total_numel % world_size
    for r in range(world_size):
        fsdp2_rank_sizes[r] = base_shard + (1 if r < rem else 0)

    z3_bytes = zero3_rank_sizes * bytes_per_elem
    fs2_bytes = fsdp2_rank_sizes * bytes_per_elem

    return {
        "zero3_bytes_per_rank": z3_bytes.tolist(),
        "fsdp2_bytes_per_rank": fs2_bytes.tolist(),
        "zero3_variance": float(np.var(z3_bytes)),
        "fsdp2_variance": float(np.var(fs2_bytes)),
        "max_imbalance_ratio": float(np.max(z3_bytes) / np.max(fs2_bytes)) if np.max(fs2_bytes) > 0 else 1.0,
    }
