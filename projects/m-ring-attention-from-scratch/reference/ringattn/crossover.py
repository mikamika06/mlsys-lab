import numpy as np


def compute_crossover(sequence_lengths, hidden_size, world_size, bandwidth, latency):
    crossovers = []
    for seq_len in sequence_lengths:
        ring_vol = 2.0 * (world_size - 1) * seq_len * hidden_size
        ulysses_vol = 2.0 * ((world_size - 1) / world_size) * seq_len * hidden_size

        ring_time = latency * (world_size - 1) + ring_vol / bandwidth
        ulysses_time = latency * 2.0 + ulysses_vol / bandwidth

        if ring_time < ulysses_time:
            crossovers.append("ring")
        else:
            crossovers.append("ulysses")
    return crossovers
