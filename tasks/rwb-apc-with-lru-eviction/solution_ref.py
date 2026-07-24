from collections import OrderedDict

def lru_apc(requests, pool_capacity):
    """Simulate an APC with LRU eviction.

    Args:
        requests: list of int, block chain hashes in arrival order.
        pool_capacity: int, maximum number of blocks in the pool.

    Returns:
        (hit_count, evicted_order) — total hits and list of evicted hashes
        in eviction order.
    """
    pool = OrderedDict()
    hit_count = 0
    evicted_order = []

    for block_hash in requests:
        if block_hash in pool:
            pool.move_to_end(block_hash)
            hit_count += 1
        else:
            if len(pool) >= pool_capacity:
                evicted_hash, _ = pool.popitem(last=False)
                evicted_order.append(evicted_hash)
            pool[block_hash] = True

    return hit_count, evicted_order
