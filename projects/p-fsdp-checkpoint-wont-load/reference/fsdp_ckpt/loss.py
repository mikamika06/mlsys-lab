import numpy as np
from fsdp_ckpt.parser import extract_chunks, align_shapes
from fsdp_ckpt.converter import consolidate


def compute_sharded_loss(sharded_checkpoints, metadata, inputs):
    """Simulate loss on sharded checkpoints to verify correctness."""
    chunks = extract_chunks(sharded_checkpoints)
    aligned = align_shapes(chunks, metadata)
    weights = consolidate(aligned, metadata)
    loss = 0.0
    for k, w in weights.items():
        loss += float(np.sum(w * inputs[k]))
    return loss
