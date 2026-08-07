import numpy as np


def ulysses_reshuffle(x, world_size):
    """Perform Ulysses All-to-All reshuffle."""
    b, s, h, d = x.shape
    s_sub = s // world_size
    h_sub = h // world_size
    reshaped = x.reshape(b, s_sub, world_size, h_sub, world_size, d)
    transposed = np.transpose(reshaped, (0, 4, 1, 2, 3, 5))
    return transposed.reshape(b, world_size * s_sub, h, d)
