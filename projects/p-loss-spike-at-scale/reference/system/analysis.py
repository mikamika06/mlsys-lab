import numpy as np

def find_first_spike(losses: list[float]) -> int:
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] * 10:
            return i
    return -1

def simulate_data_invariant(total_samples: int, num_workers: int) -> float:
    data = np.ones(total_samples, dtype=np.float32)
    true_sum = np.sum(data)
    shards = np.array_split(data, num_workers)
    sharded_sum = np.sum([np.sum(s) for s in shards])
    return float(np.abs(true_sum - sharded_sum))

def check_determinism(reduce_fn, tensors: list[np.ndarray]) -> float:
    res1 = reduce_fn(tensors)
    res2 = reduce_fn(tensors[::-1])
    return float(np.max(np.abs(res1 - res2)))
