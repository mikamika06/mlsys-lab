import numpy as np


def consolidate(aligned, metadata):
    """Reconstruct the full parameter arrays."""
    out = {}
    for k, (chunk_list, unpadded_len) in aligned.items():
        flat = np.concatenate(chunk_list)
        flat = flat[:unpadded_len]
        out[k] = flat.reshape(metadata[k])
    return out


def shard_checkpoint(consolidated, num_ranks):
    """Split parameters for a specific number of ranks."""
    ranks = [{} for _ in range(num_ranks)]
    for k, tensor in consolidated.items():
        flat = tensor.flatten()
        unpadded_len = len(flat)
        pad_len = (num_ranks - (unpadded_len % num_ranks)) % num_ranks
        if pad_len > 0:
            flat = np.concatenate([flat, np.zeros(pad_len, dtype=flat.dtype)])
        chunk_size = len(flat) // num_ranks
        for i in range(num_ranks):
            ranks[i][k] = flat[i * chunk_size : (i + 1) * chunk_size]
    return ranks
