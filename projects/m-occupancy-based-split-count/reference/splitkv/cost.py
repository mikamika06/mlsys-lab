def model_reduction_cost(batch_size, num_heads, kv_len, block_size, head_dim, num_sms, split_count, t_flop=1e-12, t_bw=1e-11, t_red=1e-10):
    num_tiles = max(1, (kv_len + block_size - 1) // block_size)
    split_count = max(1, min(split_count, num_tiles))
    total_heads = batch_size * num_heads
    total_tasks = total_heads * split_count
    waves = (total_tasks + num_sms - 1) // num_sms

    tokens_per_split = kv_len / split_count
    flops_per_task = 4.0 * tokens_per_split * head_dim
    compute_time = waves * flops_per_task * t_flop

    kv_bytes = 2.0 * total_heads * kv_len * head_dim * 2.0
    memory_time = kv_bytes * t_bw

    reduction_bytes = total_heads * split_count * (head_dim + 2) * 4.0
    reduction_time = reduction_bytes * t_red

    total_cost = compute_time + memory_time + reduction_time
    return {
        "compute_cost": compute_time,
        "memory_cost": memory_time,
        "reduction_cost": reduction_time,
        "total_cost": total_cost,
    }


def find_optimal_splits(batch_size, num_heads, kv_len, block_size, head_dim, num_sms, max_splits=128, t_flop=1e-12, t_bw=1e-11, t_red=1e-10):
    num_tiles = max(1, (kv_len + block_size - 1) // block_size)
    upper_bound = min(num_tiles, max_splits)
    best_split = 1
    best_cost = float("inf")

    for s in range(1, upper_bound + 1):
        c = model_reduction_cost(batch_size, num_heads, kv_len, block_size, head_dim, num_sms, s, t_flop, t_bw, t_red)["total_cost"]
        if c < best_cost:
            best_cost = c
            best_split = s
    return best_split
