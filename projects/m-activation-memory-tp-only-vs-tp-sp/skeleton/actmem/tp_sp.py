import numpy as np


class ToyDistributedContext:
    """Simulates a ring/mesh of ranks for distributed collective ops."""

    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.total_bytes_transferred = 0

    def all_gather(self, tensor, axis=0):
        raise NotImplementedError

    def reduce_scatter(self, tensor, axis=0):
        raise NotImplementedError

    def all_reduce(self, tensor):
        raise NotImplementedError


def simulate_sp_forward(x_shard, w_col, w_row, ctx):
    """Simulates Megatron TP+SP forward pass given local sequence shard."""
    raise NotImplementedError
