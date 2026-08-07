import numpy as np


def ulysses_all_to_all(input_shards, world_size):
    batch, seq_chunk, heads, head_dim = input_shards[0].shape
    full_seq = np.concatenate(input_shards, axis=1)
    full_heads = np.reshape(full_seq, (batch, world_size, seq_chunk, world_size, heads // world_size, head_dim))
    transposed = np.transpose(full_heads, (0, 3, 2, 1, 4, 5))
    output_shards = []
    for r in range(world_size):
        shard = transposed[:, r, :, :, :, :]
        output_shards.append(np.reshape(shard, (batch, seq_chunk, heads // world_size, head_dim)))
    return output_shards
