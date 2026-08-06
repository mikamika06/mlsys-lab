def assign_bucket(length, bucket_boundaries):
    """Assign sequence length to nearest bucket boundary."""
    raise NotImplementedError


def pad_batch(sequences, bucket_boundaries, pad_val=0):
    """Pad batch of sequence lists to assigned bucket shape."""
    raise NotImplementedError
