import numpy as np


def compute_crossover_volume(seq_len, hidden_dim, num_heads, world_size):
    ring_vol = 2.0 * (world_size - 1) * (seq_len // world_size) * hidden_dim
    ulysses_vol = 2.0 * (world_size - 1) * (seq_len * hidden_dim) / world_size
    return {"ring": ring_vol, "ulysses": ulysses_vol, "crossover": ring_vol <= ulysses_vol}
