import numpy as np
from striped.policy import assign_blocks


def simulate_throughput(num_blocks, world_size, compute_cost, comm_cost):
    assignment = assign_blocks(num_blocks, world_size)
    max_blocks = max(len(blocks) for blocks in assignment)
    total_compute = max_blocks * compute_cost
    total_comm = (world_size - 1) * comm_cost
    simulated_time = max(total_compute, total_comm)
    baseline_time = num_blocks * compute_cost + (world_size - 1) * comm_cost
    ratio = baseline_time / max(simulated_time, 1e-6)
    return float(ratio)
