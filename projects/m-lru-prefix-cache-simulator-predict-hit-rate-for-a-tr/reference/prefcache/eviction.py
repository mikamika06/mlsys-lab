from collections import OrderedDict

def reproduce_eviction_sequence(operations, capacity):
    cache = OrderedDict()
    evictions = []
    for op, block in operations:
        if op == "access":
            if block in cache:
                cache.move_to_end(block)
            else:
                if len(cache) >= capacity:
                    evicted_block, _ = cache.popitem(last=False)
                    evictions.append(evicted_block)
                cache[block] = True
        elif op == "free_reverse":
            if block in cache:
                del cache[block]
                evictions.append(block)
    return evictions
