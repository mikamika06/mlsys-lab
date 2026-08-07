def bucket_shape(seq_len, bucket_sizes):
    """Map sequence length to the smallest fitting bucket."""
    for b in sorted(bucket_sizes):
        if seq_len <= b:
            return b
    return bucket_sizes[-1]
