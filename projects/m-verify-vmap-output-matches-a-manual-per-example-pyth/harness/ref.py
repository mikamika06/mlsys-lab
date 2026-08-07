import numpy as np


def per_example_loop(fn, x_batch, axis=0):
    slices = np.swapaxes(x_batch, 0, axis)
    outs = [fn(slices[i]) for i in range(slices.shape[0])]
    res = np.stack(outs, axis=0)
    if axis != 0:
        res = np.swapaxes(res, 0, axis)
    return res


def verify_vmap_matches(fn_single, fn_batched, x_batch, axis=0):
    ref_out = per_example_loop(fn_single, x_batch, axis=axis)
    vmap_out = fn_batched(x_batch)
    return float(np.max(np.abs(ref_out - vmap_out)))


def benchmark_vmap_speedup(fn_single, fn_batched, x_batches, axis=0, timer=None):
    if timer is None:
        import time

        timer = time.perf_counter
    results = {}
    for x in x_batches:
        b_size = x.shape[axis]
        t0 = timer()
        _ = per_example_loop(fn_single, x, axis=axis)
        t1 = timer()
        t_loop = max(t1 - t0, 1e-9)

        t2 = timer()
        _ = fn_batched(x)
        t3 = timer()
        t_vmap = max(t3 - t2, 1e-9)

        speedup = t_loop / t_vmap
        results[b_size] = {
            "loop_time": float(t_loop),
            "vmap_time": float(t_vmap),
            "speedup": float(speedup),
        }
    return results


def simulated_psum(shards):
    total = np.sum(shards, axis=0)
    return [np.copy(total) for _ in shards]


def simulated_pmap(fn, x_batched, num_devices=4):
    if x_batched.shape[0] % num_devices != 0:
        raise ValueError("Batch size must be divisible by num_devices.")
    shards = np.split(x_batched, num_devices, axis=0)
    out_shards = [fn(shard) for shard in shards]
    return np.concatenate(out_shards, axis=0)


def spmd_allreduce_step(x_batched, compute_fn, num_devices=4):
    if x_batched.shape[0] % num_devices != 0:
        raise ValueError("Batch size must be divisible by num_devices.")
    shards = np.split(x_batched, num_devices, axis=0)
    local_outputs = [compute_fn(shard) for shard in shards]
    all_reduced = simulated_psum(local_outputs)
    return np.stack(all_reduced, axis=0)


def _fn1_single(x):
    return x * 2.0 + 0.5


def _fn1_batched(x):
    return x * 2.0 + 0.5


def _fn2_single(x):
    return np.sin(x) + np.exp(-np.abs(x))


def _fn2_batched(x):
    return np.sin(x) + np.exp(-np.abs(x))


def _fn3_single(x):
    return np.array([np.sum(x), np.mean(x)])


def _fn3_batched(x):
    return np.stack([np.sum(x, axis=-1), np.mean(x, axis=-1)], axis=-1)


def _fn4_single(x):
    return np.maximum(x, 0.0)


def _fn4_batched(x):
    return np.maximum(x, 0.0)


def _fn5_single(x):
    return x - np.mean(x)


def _fn5_batched(x):
    return x - np.mean(x, axis=-1, keepdims=True)


TEST_FUNCS = [
    (_fn1_single, _fn1_batched),
    (_fn2_single, _fn2_batched),
    (_fn3_single, _fn3_batched),
    (_fn4_single, _fn4_batched),
    (_fn5_single, _fn5_batched),
]

rng = np.random.RandomState(42)
TEST_BATCHES = [
    rng.randn(4, 8).astype(np.float32),
    rng.randn(8, 8).astype(np.float32),
    rng.randn(16, 8).astype(np.float32),
    rng.randn(32, 8).astype(np.float32),
]
