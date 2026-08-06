import numpy as np


def assign_bucket(length, bucket_boundaries):
    sorted_bounds = sorted(bucket_boundaries)
    for b in sorted_bounds:
        if length <= b:
            return b
    return max(sorted_bounds)


def pad_batch(sequences, bucket_boundaries, pad_val=0):
    max_len = max(len(seq) for seq in sequences)
    bucket = assign_bucket(max_len, bucket_boundaries)
    batch_size = len(sequences)
    padded = np.full((batch_size, bucket), pad_val, dtype=np.int64)
    mask = np.zeros((batch_size, bucket), dtype=np.int64)
    for i, seq in enumerate(sequences):
        length = min(len(seq), bucket)
        padded[i, :length] = seq[:length]
        mask[i, :length] = 1
    return padded, mask, bucket
