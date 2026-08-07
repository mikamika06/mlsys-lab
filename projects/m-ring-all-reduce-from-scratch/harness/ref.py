import numpy as np


def reference_ring_all_reduce(tensors, rank, world_size):
    arrs = [np.array(t, dtype=np.float32, copy=True) for t in tensors]
    n = len(arrs)
    local_results = []
    for idx, x in enumerate(arrs):
        size = x.size
        padded_size = ((size + world_size - 1) // world_size) * world_size
        padded = np.zeros(padded_size, dtype=np.float32)
        padded[:size] = x
        chunk_size = padded_size // world_size
        chunks = [padded[i * chunk_size:(i + 1) * chunk_size] for i in range(world_size)]

        recv_buffer = [np.zeros_like(chunks[0]) for _ in range(world_size)]
        for i in range(world_size):
            recv_buffer[i][:] = chunks[i]

        for step in range(world_size - 1):
            send_idx = (rank - step + world_size) % world_size
            recv_idx = (rank - step - 1 + world_size) % world_size
            send_chunk = recv_buffer[send_idx]
            incoming = send_chunk.copy()
            recv_buffer[recv_idx] += incoming

        reduced_chunks = [np.zeros_like(chunks[0]) for _ in range(world_size)]
        for i in range(world_size):
            reduced_chunks[i][:] = recv_buffer[i]

        for step in range(world_size - 1):
            send_idx = (rank - step + 1 + world_size) % world_size
            recv_idx = (rank - step + world_size) % world_size
            send_chunk = reduced_chunks[send_idx]
            incoming = send_chunk.copy()
            reduced_chunks[recv_idx] = incoming

        reassembled = np.concatenate(reduced_chunks)[:size]
        local_results.append(reassembled)
    return local_results


def reference_ring_cost(size_bytes, world_size, alpha, beta):
    return 2.0 * (world_size - 1) * alpha + (2.0 * (world_size - 1) / world_size) * size_bytes * beta


def reference_tree_cost(size_bytes, world_size, alpha, beta):
    import math
    steps = 2.0 * math.log2(world_size)
    return steps * alpha + steps * size_bytes * beta


def reference_find_crossover(world_size, alpha, beta):
    low = 1
    high = 100_000_000
    for _ in range(50):
        mid = (low + high) / 2.0
        rc = reference_ring_cost(mid, world_size, alpha, beta)
        tc = reference_tree_cost(mid, world_size, alpha, beta)
        if rc < tc:
            low = mid
        else:
            high = mid
    return float(low)
