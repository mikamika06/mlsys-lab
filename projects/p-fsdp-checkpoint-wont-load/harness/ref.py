import numpy as np


def extract_chunks(checkpoints):
    """Reference implementation of parsing chunks."""
    if not checkpoints:
        return {}
    out = {}
    for k in checkpoints[0].keys():
        out[k] = [ckpt[k] for ckpt in checkpoints]
    return out


def align_shapes(chunks, metadata):
    """Reference implementation of align shapes."""
    out = {}
    for k, chunk_list in chunks.items():
        target_shape = metadata[k]
        unpadded_len = int(np.prod(target_shape))
        out[k] = (chunk_list, unpadded_len)
    return out


def consolidate(aligned, metadata):
    """Reference implementation of consolidate."""
    out = {}
    for k, (chunk_list, unpadded_len) in aligned.items():
        flat = np.concatenate(chunk_list)
        flat = flat[:unpadded_len]
        out[k] = flat.reshape(metadata[k])
    return out


def shard_checkpoint(consolidated, num_ranks):
    """Reference implementation of sharding."""
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


def compute_sharded_loss(sharded_checkpoints, metadata, inputs):
    """Reference implementation of loss."""
    chunks = extract_chunks(sharded_checkpoints)
    aligned = align_shapes(chunks, metadata)
    weights = consolidate(aligned, metadata)
    loss = 0.0
    for k, w in weights.items():
        loss += float(np.sum(w * inputs[k]))
    return loss


def get_metadata():
    """Returns fixed mock metadata for tests."""
    return {"l1.weight": (64, 64), "l1.bias": (64,), "l2.weight": (32, 64)}


def get_consolidated():
    """Returns a deterministic consolidated checkpoint."""
    np.random.seed(42)
    meta = get_metadata()
    return {k: np.random.randn(*shape) for k, shape in meta.items()}


def get_fixture_ckpt(ranks):
    """Returns a sharded mock checkpoint for the given number of ranks."""
    return shard_checkpoint(get_consolidated(), ranks)
