def extract_chunks(checkpoints):
    """Group sharded parameters across all ranks."""
    raise NotImplementedError


def align_shapes(chunks, metadata):
    """Calculate unpadded length for each parameter."""
    raise NotImplementedError
