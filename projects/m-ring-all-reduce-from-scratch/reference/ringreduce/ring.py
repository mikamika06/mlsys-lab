import numpy as np


def ring_all_reduce(tensors, rank, world_size):
    arrs = [np.array(t, dtype=np.float32, copy=True) for t in tensors]
    num_chunks = world_size
    chunk_size = len(arrs[rank]) // num_chunks

    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        send_chunk_start = send_idx * chunk_size
        send_chunk_end = (send_idx + 1) * chunk_size if send_idx != world_size - 1 else len(arrs[rank])

        recv_chunk_start = recv_idx * chunk_size
        recv_chunk_end = (recv_idx + 1) * chunk_size if recv_idx != world_size - 1 else len(arrs[rank])

        pass_data = arrs[rank][send_chunk_start:send_chunk_end].copy()
        prev_rank = (rank - 1 + world_size) % world_size

        arrs[prev_rank][send_chunk_start:send_chunk_end] = pass_data
        arrs[rank][recv_chunk_start:recv_chunk_end] += arrs[prev_rank][recv_chunk_start:recv_chunk_end]

    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        send_chunk_start = send_idx * chunk_size
        send_chunk_end = (send_idx + 1) * chunk_size if send_idx != world_size - 1 else len(arrs[rank])

        recv_chunk_start = recv_idx * chunk_size
        recv_chunk_end = (recv_idx + 1) * chunk_size if recv_idx != world_size - 1 else len(arrs[rank])

        pass_data = arrs[rank][send_chunk_start:send_chunk_end].copy()
        prev_rank = (rank - 1 + world_size) % world_size

        arrs[prev_rank][send_chunk_start:send_chunk_end] = pass_data

    return arrs[rank]
