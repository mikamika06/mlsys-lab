def lru_apc(requests, pool_capacity):
    """Simulate an APC with LRU eviction.

    Args:
        requests: list of int, block chain hashes in arrival order.
        pool_capacity: int, maximum number of blocks in the pool.

    Returns:
        (hit_count, evicted_order) — total hits and list of evicted hashes
        in eviction order.
    """
    raise NotImplementedError('your code here')
