def communication_volume(num_tokens, hidden_dim, num_experts, top_k, world_size, capacity, bytes_per_elem=4):
    actual_dispatched = min(num_tokens * top_k, num_experts * capacity)
    moe_dispatch_bytes = actual_dispatched * hidden_dim * bytes_per_elem
    moe_combine_bytes = moe_dispatch_bytes
    moe_total_bytes = moe_dispatch_bytes + moe_combine_bytes

    dense_allreduce_bytes = 2 * (world_size - 1) / world_size * (num_tokens * hidden_dim) * bytes_per_elem

    return {
        "moe_total_bytes": int(moe_total_bytes),
        "dense_total_bytes": int(dense_allreduce_bytes),
        "ratio_moe_to_dense": float(moe_total_bytes / dense_allreduce_bytes) if dense_allreduce_bytes > 0 else 0.0
    }
