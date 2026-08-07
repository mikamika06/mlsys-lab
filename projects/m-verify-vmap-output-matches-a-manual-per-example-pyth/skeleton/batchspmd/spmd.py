def simulated_psum(shards):
    """Performs elementwise all-reduce sum across device shards."""
    raise NotImplementedError


def simulated_pmap(fn, x_batched, num_devices=4):
    """Splits x_batched into num_devices shards, applies fn to each shard, and concatenates results."""
    raise NotImplementedError


def spmd_allreduce_step(x_batched, compute_fn, num_devices=4):
    """Executes compute_fn per device shard, performs psum all-reduce across devices, and returns stacked output."""
    raise NotImplementedError
