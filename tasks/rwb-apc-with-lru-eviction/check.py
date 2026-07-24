def _ref(requests, pool_capacity):
    """Reference LRU-APC oracle using collections.OrderedDict."""
    from collections import OrderedDict
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

def grade(sol, fx) -> dict:
    cases = [
        ([1, 2, 3, 1, 2, 4], 3),
        ([1, 2, 3, 4, 1, 2, 5], 3),
        ([1, 2, 3, 1, 2, 3], 3),
        ([1, 1, 2, 2], 1),
        ([], 3),
        ([5, 5, 5, 5], 5),
        ([10, 20, 30, 20, 40, 50, 30, 60], 3),
        (list(range(20)) * 3, 5),
        ([1, 2, 1, 3, 1, 4, 1, 5], 3),
        ([7, 7, 7, 8, 8, 9, 7, 9, 10], 2),
    ]
    ok = 1.0
    for requests, capacity in cases:
        try:
            got_hit, got_evicted = sol.lru_apc(list(requests), capacity)
            ref_hit, ref_evicted = _ref(list(requests), capacity)
        except Exception:
            ok = 0.0
            break
        if got_hit != ref_hit or list(got_evicted) != list(ref_evicted):
            ok = 0.0
            break
    return {"exact_match": ok}
