import numpy as np


def ulysses_all_to_all(x, world_size):
    b, s_seq, h_heads, d = x.shape
    local_s = s_seq // world_size
    local_h = h_heads // world_size

    reshaped = x.reshape(b, world_size, local_s, world_size, local_h, d)
    transposed = np.transpose(reshaped, (0, 3, 2, 1, 4, 5))
    output = transposed.reshape(b, local_s, h_heads, d)
    return output
