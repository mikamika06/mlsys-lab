import numpy as np


def ring_all_reduce(local_tensor: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    arr = np.copy(local_tensor)
    n = arr.size
    chunk_size = n // world_size
    tensors = [np.copy(arr) for _ in range(world_size)]
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size
        send_start = send_idx * chunk_size
        send_end = (send_idx + 1) * chunk_size if send_idx != world_size - 1 else n
        recv_start = recv_idx * chunk_size
        recv_end = (recv_idx + 1) * chunk_size if recv_idx != world_size - 1 else n
        val = tensors[send_idx][send_start:send_end]
        tensors[recv_idx][recv_start:recv_end] += val
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size
        send_start = send_idx * chunk_size
        send_end = (send_idx + 1) * chunk_size if send_idx != world_size - 1 else n
        recv_start = recv_idx * chunk_size
        recv_end = (recv_idx + 1) * chunk_size if recv_idx != world_size - 1 else n
        tensors[recv_idx][recv_start:recv_end] = tensors[send_idx][send_start:send_end]
    return tensors[rank]
