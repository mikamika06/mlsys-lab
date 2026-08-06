def simulate_sdpa_pass(seq_len):
    base_ops = 3 * seq_len
    fused_ops = int(seq_len * 0.4) + 2
    memory_unfused = seq_len * 1024
    memory_fused = int(seq_len * 512)
    return {
        "ops_before": base_ops,
        "ops_after": fused_ops,
        "memory_bytes_before": memory_unfused,
        "memory_bytes_after": memory_fused,
        "fused": True
    }
