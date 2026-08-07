import numpy as np


def simulated_psum(shards):
    """Performs elementwise all-reduce sum across device shards."""
    total = np.sum(shards, axis=0)
    return [np.copy(total) for _ in shards]


def simulated_pmap(fn, x_batched, num_devices=4):
    """Splits x_batched into num_devices shards, applies fn to each shard, and concatenates results."""
    if x_batched.shape[0] % num_devices != 0:
        raise ValueError("Batch size must be divisible by num_devices.")
    shards = np.split(x_batched, num_devices, axis=0)
    out_shards = [fn(shard) for shard in shards]
    return np.concatenate(out_shards, axis=0)


def spmd_allreduce_step(x_batched, compute_fn, num_devices=4):
    """Executes compute_fn per device shard, performs psum all-reduce across devices, and returns stacked output."""
    if x_batched.shape[0] % num_devices != 0:
        raise ValueError("Batch size must be divisible by num_devices.")
    shards = np.split(x_batched, num_devices, axis=0)
    local_outputs = [compute_fn(shard) for shard in shards]
    all_reduced = simulated_psum(local_outputs)
    return np.stack(all_reduced, axis=0)
