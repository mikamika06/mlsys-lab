import numpy as np


def compute_valid_pairs_causal(q_indices: np.ndarray, k_indices: np.ndarray) -> int:
    q = np.asarray(q_indices, dtype=np.int64)[:, None]
    k = np.asarray(k_indices, dtype=np.int64)[None, :]
    return int(np.sum(q >= k))


def assign_chunks_naive(num_chunks: int, world_size: int) -> np.ndarray:
    chunks_per_device = num_chunks // world_size
    return np.repeat(np.arange(world_size), chunks_per_device)


def assign_chunks_striped(num_chunks: int, world_size: int) -> np.ndarray:
    return np.arange(num_chunks) % world_size


def assign_chunks_zigzag(num_chunks: int, world_size: int) -> np.ndarray:
    device_assignments = np.zeros(num_chunks, dtype=np.int64)
    chunks_per_device = num_chunks // world_size
    for c in range(chunks_per_device):
        start_idx = c * world_size
        end_idx = (c + 1) * world_size
        if c % 2 == 0:
            device_assignments[start_idx:end_idx] = np.arange(world_size)
        else:
            device_assignments[start_idx:end_idx] = np.arange(world_size - 1, -1, -1)
    return device_assignments
