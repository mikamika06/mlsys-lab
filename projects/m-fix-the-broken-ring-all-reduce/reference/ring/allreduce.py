import numpy as np

def ring_all_reduce(tensors):
    world_size = len(tensors)
    data = [np.copy(t).astype(np.float32) for t in tensors]
    out = [np.copy(d) for d in data]
    for step in range(world_size - 1):
        send_data = [np.copy(d) for d in out]
        for r in range(world_size):
            next_r = (r + 1) % world_size
            prev_r = (r - 1 + world_size) % world_size
            out[next_r] += send_data[r]
    return out
