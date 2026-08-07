import numpy as np

def compute_crossover_seq_len(hidden_dim, num_heads, world_size, block_size):
    ring_comm = 2 * (world_size - 1) * hidden_dim
    ulysses_comm = 2 * (world_size - 1) * hidden_dim / world_size
    crossover = (ring_comm + ulysses_comm) / (hidden_dim + 1.0) * float(block_size)
    return float(max(128.0, crossover))
