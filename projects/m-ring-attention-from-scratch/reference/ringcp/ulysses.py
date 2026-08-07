import numpy as np

def ulysses_all_to_all(tensor, world_size, rank, scatter_dim, gather_dim):
    shape = tensor.shape
    chunks = np.array_split(tensor, world_size, axis=scatter_dim)
    received_chunks = []
    for r in range(world_size):
        send_idx = (rank + r) % world_size
        recv_idx = (rank - r + world_size) % world_size
        rc = np.array_split(chunks[send_idx], world_size, axis=gather_dim)[recv_idx]
        received_chunks.append(rc)

    gathered = []
    for i in range(world_size):
        sub_chunks = []
        for r in range(world_size):
            send_idx = (r) % world_size
            recv_idx = (r - i + world_size) % world_size
            c = np.array_split(np.array_split(tensor, world_size, axis=scatter_dim)[send_idx], world_size, axis=gather_dim)[recv_idx]
            sub_chunks.append(c)
        gathered.append(np.concatenate(sub_chunks, axis=gather_dim))

    return gathered[rank]
