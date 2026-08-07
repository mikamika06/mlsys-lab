def compute_crossover(seq_len, hidden_size, world_size, num_heads):
    ring_vol = 2 * (world_size - 1) * seq_len * (hidden_size // world_size) * 4
    ulysses_vol = 2 * ((world_size - 1) / world_size) * seq_len * hidden_size * 4
    return ring_vol, ulysses_vol
