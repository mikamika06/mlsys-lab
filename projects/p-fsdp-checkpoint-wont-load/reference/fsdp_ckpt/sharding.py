"""Sharding specifications and tensor reconstruction logic."""

import math
import numpy as np


def calculate_shard_spec(global_shape, world_size, rank):
    """Calculate shard shape, offset, and padding for dim 0 sharding."""
    dim0 = global_shape[0]
    chunk_size = math.ceil(dim0 / world_size)
    start_offset = rank * chunk_size
    end_offset = min((rank + 1) * chunk_size, dim0)

    if start_offset >= dim0:
        actual_len = 0
        padding = chunk_size
    else:
        actual_len = end_offset - start_offset
        padding = chunk_size - actual_len

    shard_shape = (chunk_size,) + tuple(global_shape[1:])
    return {
        "shard_shape": shard_shape,
        "shard_offset": start_offset,
        "padding": padding,
    }


def reconstruct_param(rank_shards_info):
    """Reconstruct a global parameter array from per-rank shards."""
    sorted_shards = sorted(rank_shards_info, key=lambda x: x[1]["shard_offset"])
    clean_pieces = []

    for tensor, meta in sorted_shards:
        padding = meta.get("padding", 0)
        if padding > 0:
            clean_slice = tensor[:-padding] if padding < tensor.shape[0] else tensor[:0]
        else:
            clean_slice = tensor
        if clean_slice.shape[0] > 0:
            clean_pieces.append(clean_slice)

    if not clean_pieces:
        global_shape = sorted_shards[0][1]["global_shape"]
        return np.zeros(global_shape)

    reconstructed = np.concatenate(clean_pieces, axis=0)
    expected_shape = tuple(sorted_shards[0][1]["global_shape"])
    return reconstructed.reshape(expected_shape)
