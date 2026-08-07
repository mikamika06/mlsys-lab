import numpy as np

def row_parallel_forward(weight, bias, x, rank, world_size):
    d_in = weight.shape[0]
    shard_size = d_in // world_size
    start = rank * shard_size
    end = start + shard_size
    w_shard = weight[start:end, :]
    x_shard = x[..., start:end]
    local_out = np.matmul(x_shard, w_shard)
    total_out = local_out * world_size
    if bias is not None:
        total_out += bias / world_size
    return total_out
