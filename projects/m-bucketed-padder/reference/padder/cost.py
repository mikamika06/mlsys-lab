from padder.bucket import assign_bucket


def compute_padding_waste(lengths, bucket_boundaries):
    sorted_bounds = sorted(bucket_boundaries)
    total_unpadded = sum(lengths)
    total_padded = sum(assign_bucket(l, sorted_bounds) for l in lengths)
    waste_tokens = total_padded - total_unpadded
    ratio = waste_tokens / float(total_padded) if total_padded > 0 else 0.0
    return waste_tokens, ratio
