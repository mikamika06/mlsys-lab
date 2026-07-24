def round_to_bucket(captured_sizes: list[int], batch_sizes: list[int]) -> tuple[list[int], list[int]]:
    """
    For each requested batch size, find the smallest captured bucket that is at least as large.
    If no such bucket exists, mark the batch as eager with -1 and zero padding.
    Returns two lists of equal length: chosen buckets and padded rows.
    """
    chosen = []
    padded = []
    for b in batch_sizes:
        # Find first captured size >= b
        bucket = next((c for c in captured_sizes if c >= b), None)
        if bucket is None:
            chosen.append(-1)
            padded.append(0)
        else:
            chosen.append(bucket)
            padded.append(bucket - b)
    return chosen, padded
