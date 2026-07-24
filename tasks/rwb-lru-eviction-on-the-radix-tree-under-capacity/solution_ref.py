def radix_lru_cache(trace, capacity, followup):
    nodes = {}        # prefix tuple -> last_touch clock value
    child_count = {}  # prefix tuple -> number of live children
    clock = 0
    evicted_total = 0

    def touch_path(seq):
        nonlocal clock
        prefix = ()
        for tok in seq:
            prefix = prefix + (tok,)
            clock += 1
            if prefix not in nodes:
                parent = prefix[:-1]
                child_count[parent] = child_count.get(parent, 0) + 1
                child_count.setdefault(prefix, 0)
            nodes[prefix] = clock

    def evict_to_capacity():
        nonlocal evicted_total
        while len(nodes) > capacity:
            leaf = None
            best_t = None
            for p, t in nodes.items():
                if child_count.get(p, 0) == 0:
                    if best_t is None or t < best_t:
                        best_t = t
                        leaf = p
            del nodes[leaf]
            child_count.pop(leaf, None)
            parent = leaf[:-1]
            child_count[parent] -= 1
            evicted_total += 1

    for _op, seq in trace:
        touch_path(tuple(seq))
        evict_to_capacity()

    hits = sum(1 for seq in followup if tuple(seq) in nodes)
    hit_rate = hits / len(followup) if followup else 0.0

    return evicted_total, hit_rate
