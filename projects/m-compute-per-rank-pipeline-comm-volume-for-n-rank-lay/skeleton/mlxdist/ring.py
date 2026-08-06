import numpy as np


def run_ring_all_reduce(rank, world_size, tensor):
    """Simulate ring all-reduce execution across ring ranks."""
    raise NotImplementedError


def launch_2rank_ring_all_reduce(tensor_a, tensor_b):
    """Launch 2 local ring ranks doing all_reduce."""
    raise NotImplementedError
