import numpy as np


def extract_chunks(checkpoints):
    """Group sharded parameters across all ranks."""
    if not checkpoints:
        return {}
    out = {}
    for k in checkpoints[0].keys():
        out[k] = [ckpt[k] for ckpt in checkpoints]
    return out


def align_shapes(chunks, metadata):
    """Calculate unpadded length for each parameter."""
    out = {}
    for k, chunk_list in chunks.items():
        target_shape = metadata[k]
        unpadded_len = int(np.prod(target_shape))
        out[k] = (chunk_list, unpadded_len)
    return out
