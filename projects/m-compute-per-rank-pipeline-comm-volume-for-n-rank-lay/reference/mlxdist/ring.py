import numpy as np


def run_ring_all_reduce(rank, world_size, tensor):
    """Simulate ring all-reduce execution across ring ranks."""
    arr = np.array(tensor, dtype=np.float32)
    chunk_size = len(arr) // world_size
    chunks = [
        arr[i * chunk_size : (i + 1) * chunk_size].copy()
        for i in range(world_size)
    ]
    return chunks


def launch_2rank_ring_all_reduce(tensor_a, tensor_b):
    """Launch 2 local ring ranks doing all_reduce."""
    a = np.array(tensor_a, dtype=np.float32)
    b = np.array(tensor_b, dtype=np.float32)

    res_a = np.copy(a)
    res_b = np.copy(b)

    half = len(a) // 2

    chunk0_sum = a[:half] + b[:half]
    chunk1_sum = a[half:] + b[half:]

    res_a[:half] = chunk0_sum
    res_a[half:] = chunk1_sum
    res_b[:half] = chunk0_sum
    res_b[half:] = chunk1_sum

    return {"rank0": res_a, "rank1": res_b}
