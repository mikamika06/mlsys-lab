def count_transactions(arr, stride):
    """
    Count the number of 128‑byte memory transactions for a warp that reads
    elements spaced by `stride` starting at index 0.
    """
    segments = set()
    for t in range(32):
        idx = t * stride
        seg = idx // 32
        segments.add(seg)
    return len(segments)
