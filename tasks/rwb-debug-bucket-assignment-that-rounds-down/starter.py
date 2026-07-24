def bucket_assign(sizes: list[int], buckets: list[int]) -> list[int]:
    """Assign each size to the bucket with nearest capacity (BUGGY)."""
    indices = []
    for s in sizes:
        i = min(range(len(buckets)), key=lambda i: abs(buckets[i] - s))
        indices.append(i)
    return indices
