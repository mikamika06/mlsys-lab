def radix_lru_cache(trace, capacity, followup):
    """Simulate a radix-tree prefix cache with LRU-over-leaves eviction.

    trace: list of ("insert", seq) / ("query", seq) ops, in order. Both
        kinds walk seq's prefixes root-to-leaf, creating any missing
        node, and tick a global clock once per node touched (existing or
        new), stamping that node's last_touch.
    capacity: max number of tree nodes (tokens) allowed at any time.
    followup: sequences to check after the trace finishes.

    After every op, while the tree has more than `capacity` nodes,
    repeatedly evict the current leaf (a node with no live children)
    with the smallest last_touch value, one node per eviction step.

    Returns (evicted_token_total, hit_rate):
      evicted_token_total: total nodes evicted across the whole trace.
      hit_rate: fraction of `followup` sequences whose entire path is
        still present in the tree after the trace finishes.
    """
    raise NotImplementedError('your code here')
