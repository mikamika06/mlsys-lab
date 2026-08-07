def prefix_reuse_savings(trace: list[list[int]], chunk_size: int = 512) -> int:
    """
    Reference implementation of the prefix‑chunk reuse savings algorithm.
    Uses a set of tuples to store all contiguous subsequences up to `chunk_size`.
    """
    cache = set()
    total_saved = 0
    for req in trace:
        # longest cached prefix
        max_len = 0
        n = len(req)
        limit = min(n, chunk_size)
        for l in range(1, limit + 1):
            if tuple(req[:l]) in cache and l > max_len:
                max_len = l
        total_saved += max_len

        # add all contiguous subsequences up to chunk_size
        for i in range(n):
            for l in range(1, min(chunk_size, n - i) + 1):
                cache.add(tuple(req[i:i + l]))
    return total_saved
