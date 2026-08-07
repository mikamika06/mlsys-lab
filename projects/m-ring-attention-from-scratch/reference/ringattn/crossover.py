def compute_crossover(seq_len, hidden_size, world_size):
    ring_comm = 2 * (world_size - 1) * seq_len * hidden_size * 4
    ulysses_comm = 2 * (world_size - 1) / world_size * seq_len * hidden_size * 4
    return ring_comm, ulysses_comm
