import numpy as np

def ulysses_reshuffle(x, world_size, rank, forward=True):
    b, s_p, h_p, d = x.shape
    if forward:
        s = s_p * world_size
        h = h_p * world_size
        x_reshaped = x.reshape(b, world_size, s_p // world_size, world_size, h_p, d)
        x_transposed = np.transpose(x_reshaped, (0, 2, 1, 3, 4, 5))
        x_flat = x_transposed.reshape(b, s // world_size, h, d)
        h_chunk_size = h // world_size
        return x_flat[:, :, rank * h_chunk_size : (rank + 1) * h_chunk_size, :]
    else:
        h = h_p * world_size
        h_chunk_size = h_p
        s_chunk_size = s_p
        x_gathered = np.zeros((b, s_chunk_size * world_size, h_chunk_size, d), dtype=x.dtype)
        for r in range(world_size):
            x_gathered[:, r * s_chunk_size : (r + 1) * s_chunk_size, :, :] = x
        return x_gathered
