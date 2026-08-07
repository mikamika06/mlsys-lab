import math


def partition_flat_contiguous(
    tensor_sizes: list[int],
    world_size: int,
    alignment: int = 1,
) -> dict:
    """Partition flattened contiguous parameter buffer across DP ranks with alignment."""
    total_elements = sum(tensor_sizes)
    if total_elements == 0 or world_size <= 0:
        per_rank_size = 0
    else:
        denom = world_size * alignment
        per_rank_size = math.ceil(total_elements / denom) * alignment

    padded_total = per_rank_size * world_size

    tensor_offsets = []
    curr = 0
    for sz in tensor_sizes:
        tensor_offsets.append((curr, curr + sz))
        curr += sz

    ranks = []
    for r in range(world_size):
        r_start = r * per_rank_size
        r_end = r_start + per_rank_size
        fragments = []
        for idx, (t_start, t_end) in enumerate(tensor_offsets):
            o_start = max(r_start, t_start)
            o_end = min(r_end, t_end)
            if o_start < o_end:
                fragments.append({
                    "tensor_idx": idx,
                    "tensor_offset": o_start - t_start,
                    "num_elements": o_end - o_start,
                })
        ranks.append({
            "rank": r,
            "start": r_start,
            "end": r_end,
            "fragments": fragments,
        })

    return {
        "per_rank_size": per_rank_size,
        "padded_total": padded_total,
        "ranks": ranks,
    }


def partition_bin_packing(
    tensor_sizes: list[int],
    world_size: int,
) -> dict:
    """Bin-pack whole tensors into DP ranks using Longest Processing Time (LPT) heuristic."""
    if world_size <= 0:
        return {
            "assignments": [],
            "loads": [],
            "max_load": 0,
            "min_load": 0,
            "imbalance": 0,
        }

    indexed = [(sz, idx) for idx, sz in enumerate(tensor_sizes)]
    indexed.sort(key=lambda x: (-x[0], x[1]))

    loads = [0] * world_size
    assignments = [[] for _ in range(world_size)]

    for sz, idx in indexed:
        min_val = min(loads)
        target_rank = loads.index(min_val)
        assignments[target_rank].append(idx)
        loads[target_rank] += sz

    for r in range(world_size):
        assignments[r].sort()

    max_l = max(loads) if loads else 0
    min_l = min(loads) if loads else 0

    return {
        "assignments": assignments,
        "loads": loads,
        "max_load": max_l,
        "min_load": min_l,
        "imbalance": max_l - min_l,
    }
