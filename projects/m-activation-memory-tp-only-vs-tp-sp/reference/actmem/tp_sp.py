import numpy as np


class ToyDistributedContext:
    """Simulates a ring/mesh of ranks for distributed collective ops."""

    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self.total_bytes_transferred = 0

    def all_gather(self, tensor, axis=0):
        nbytes = tensor.nbytes * (self.world_size - 1)
        self.total_bytes_transferred += nbytes
        gathered = [np.copy(tensor) for _ in range(self.world_size)]
        return np.concatenate(gathered, axis=axis)

    def reduce_scatter(self, tensor, axis=0):
        nbytes = (tensor.nbytes // self.world_size) * (self.world_size - 1)
        self.total_bytes_transferred += nbytes
        chunks = np.split(tensor, self.world_size, axis=axis)
        reduced = sum(chunks)
        return reduced

    def all_reduce(self, tensor):
        nbytes = 2 * tensor.nbytes * ((self.world_size - 1) / self.world_size)
        self.total_bytes_transferred += nbytes
        return np.copy(tensor)


def simulate_sp_forward(x_shard, w_col, w_row, ctx):
    """Simulates Megatron TP+SP forward pass given local sequence shard."""
    x_full = ctx.all_gather(x_shard, axis=0)
    h_col = np.dot(x_full, w_col)
    h_act = np.maximum(0, h_col)
    h_row = np.dot(h_act, w_row)
    out_shard = ctx.reduce_scatter(h_row, axis=0)
    return out_shard
