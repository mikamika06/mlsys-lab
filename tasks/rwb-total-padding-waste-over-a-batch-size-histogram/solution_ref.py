def total_padding_waste(size_histogram: dict[int,int], bucket_size:int) -> int:
    """
    Compute the total number of padded rows that are added when each batch in a
    workload is padded to the nearest multiple of ``bucket_size``.
    """
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    total = 0
    for b, count in size_histogram.items():
        waste = (bucket_size - (b % bucket_size)) % bucket_size
        total += count * waste
    return int(total)
