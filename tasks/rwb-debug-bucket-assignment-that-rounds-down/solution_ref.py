def bucket_assign(sizes: list[int], buckets: list[int]) -> list[int]:
    """Assign each size to the smallest bucket that is >= size (else largest)."""
    indices = []
    for s in sizes:
        idx = 0
        while idx < len(buckets) - 1 and buckets[idx] < s:
            idx += 1
        indices.append(idx)
    return indices
