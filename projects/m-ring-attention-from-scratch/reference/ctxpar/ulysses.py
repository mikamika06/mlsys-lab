import numpy as np


def ulysses_all_to_all(x, world_size, rank, seq_to_head=True):
    if seq_to_head:
        batch, seq_chunk, heads, dim = x.shape
        new_seq_len = seq_chunk // world_size
        new_heads = heads * world_size
        reshaped = x.reshape(batch, world_size, new_seq_len, heads, dim)
        transposed = np.transpose(reshaped, (0, 3, 2, 1, 4))
        out = transposed.reshape(batch, heads, new_seq_len, new_heads * dim // heads)
        return out
    else:
        batch, heads, seq_len, head_dim = x.shape
        new_heads = heads // world_size
        new_seq_len = seq_len * world_size
        reshaped = x.reshape(batch, world_size, new_heads, seq_len, head_dim)
        transposed = np.transpose(reshaped, (0, 3, 2, 1, 4))
        out = transposed.reshape(batch, new_seq_len, new_heads, head_dim)
        return out
