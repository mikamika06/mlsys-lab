def balance_lengths(samples, bucket_sizes, target_counts):
    """Group and sample sequences into target length buckets."""
    sorted_buckets = sorted(bucket_sizes)
    buckets = {b: [] for b in sorted_buckets}

    for s in samples:
        length = len(s)
        for b in sorted_buckets:
            if length <= b:
                buckets[b].append(s)
                break

    result = []
    for b in sorted_buckets:
        target = target_counts.get(b, 0)
        available = buckets[b]
        if not available:
            continue
        if len(available) >= target:
            result.extend(available[:target])
        else:
            repeated = []
            while len(repeated) < target:
                repeated.extend(available)
            result.extend(repeated[:target])

    return result
