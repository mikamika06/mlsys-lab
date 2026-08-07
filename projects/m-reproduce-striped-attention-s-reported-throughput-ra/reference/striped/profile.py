from striped.simulator import simulate_block_attention, simulate_striped_attention


def profile_workload(seq_len, block_size, world_size, strategy):
    if strategy == "block":
        return simulate_block_attention(seq_len, block_size, world_size)
    elif strategy == "striped":
        return simulate_striped_attention(seq_len, block_size, world_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
