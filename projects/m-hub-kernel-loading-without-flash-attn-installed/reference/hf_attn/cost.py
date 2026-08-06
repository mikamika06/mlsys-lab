def estimate_train_step_cost(batch_size, seq_len, num_heads, head_dim, backend="sdpa"):
    flops_fwd = 4 * batch_size * num_heads * (seq_len ** 2) * head_dim
    flops_bwd = 2.5 * flops_fwd
    total_flops = flops_fwd + flops_bwd

    overhead_multipliers = {
        "flash": 1.0,
        "sdpa": 1.25,
        "math": 2.5,
    }
    multiplier = overhead_multipliers.get(backend, 2.0)

    memory_bytes = batch_size * num_heads * seq_len * head_dim * 2 * 3
    if backend == "math":
        memory_bytes += batch_size * num_heads * (seq_len ** 2) * 4

    estimated_cost = total_flops * multiplier + memory_bytes
    return {
        "flops": total_flops,
        "memory_bytes": memory_bytes,
        "effective_cost": estimated_cost
    }


def compare_backend_costs(batch_size, seq_len, num_heads, head_dim, available_backends):
    results = {}
    for b in available_backends:
        name = b if isinstance(b, str) else b.name
        results[name] = estimate_train_step_cost(batch_size, seq_len, num_heads, head_dim, backend=name)
    return results
