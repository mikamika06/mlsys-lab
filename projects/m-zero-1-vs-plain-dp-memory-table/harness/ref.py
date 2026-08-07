import math

CONFIGS_M1 = [
    {
        "param_counts": [1024, 2048, 4096, 8192],
        "world_size": 4,
        "bytes_per_param": 2,
        "bytes_per_grad": 2,
        "opt_bytes_per_param": 12,
    },
    {
        "param_counts": [500000, 1200000, 300000],
        "world_size": 8,
        "bytes_per_param": 2,
        "bytes_per_grad": 2,
        "opt_bytes_per_param": 12,
    },
    {
        "param_counts": [100, 200],
        "world_size": 1,
        "bytes_per_param": 2,
        "bytes_per_grad": 2,
        "opt_bytes_per_param": 12,
    },
    {
        "param_counts": [10000, 20000],
        "world_size": 2,
        "bytes_per_param": 4,
        "bytes_per_grad": 4,
        "opt_bytes_per_param": 16,
    },
]

CONFIGS_M2 = [
    {
        "tensor_sizes": [100, 200, 150],
        "world_size": 2,
        "alignment": 1,
    },
    {
        "tensor_sizes": [100, 200, 150],
        "world_size": 2,
        "alignment": 8,
    },
    {
        "tensor_sizes": [512, 1024, 2048, 4096],
        "world_size": 4,
        "alignment": 16,
    },
]

CONFIGS_M3 = [
    {
        "tensor_sizes": [10, 50, 30, 20, 40],
        "world_size": 2,
    },
    {
        "tensor_sizes": [100, 200, 150, 350, 250, 500, 80],
        "world_size": 4,
    },
    {
        "tensor_sizes": [500, 500, 500, 500],
        "world_size": 2,
    },
]


def calc_memory_table(
    param_counts: list[int],
    world_size: int,
    bytes_per_param: int = 2,
    bytes_per_grad: int = 2,
    opt_bytes_per_param: int = 12,
) -> dict:
    """Calculate memory breakdown for Plain DP vs ZeRO Stage 1."""
    total_params = sum(param_counts)
    plain_p = total_params * bytes_per_param
    plain_g = total_params * bytes_per_grad
    plain_o = total_params * opt_bytes_per_param
    plain_tot = plain_p + plain_g + plain_o

    zero_p = plain_p
    zero_g = plain_g
    zero_o = math.ceil(plain_o / world_size) if world_size > 0 else 0
    zero_tot = zero_p + zero_g + zero_o

    return {
        "total_params": total_params,
        "plain_dp": {
            "params_bytes": plain_p,
            "grads_bytes": plain_g,
            "opt_bytes": plain_o,
            "total_bytes": plain_tot,
        },
        "zero1": {
            "params_bytes": zero_p,
            "grads_bytes": zero_g,
            "opt_bytes": zero_o,
            "total_bytes": zero_tot,
        },
        "opt_savings_bytes": plain_o - zero_o,
        "total_savings_bytes": plain_tot - zero_tot,
    }


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
