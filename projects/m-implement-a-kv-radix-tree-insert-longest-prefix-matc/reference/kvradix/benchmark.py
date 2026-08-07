from kvradix.eviction import EvictableRadixCache


def simulate_agent_trace(trace, cache_type, block_size=16, capacity_tokens=1024):
    total_tokens = 0
    hit_tokens = 0

    if cache_type == "radix":
        cache = EvictableRadixCache(max_tokens=capacity_tokens)
        for req in trace:
            tokens = req["tokens"]
            n = len(tokens)
            total_tokens += n
            matched_len, node = cache.insert_and_cache(tokens)
            hit_tokens += matched_len
    elif cache_type == "block_hash":
        cache = {}
        lru = {}
        clock = 0

        for req in trace:
            tokens = req["tokens"]
            n = len(tokens)
            total_tokens += n

            num_blocks = n // block_size
            matched_blocks = 0

            for i in range(num_blocks):
                block = tuple(tokens[i * block_size : (i + 1) * block_size])
                clock += 1
                if block in cache:
                    matched_blocks += 1
                    lru[block] = clock
                else:
                    break

            hit_tokens += matched_blocks * block_size

            for i in range(num_blocks):
                block = tuple(tokens[i * block_size : (i + 1) * block_size])
                clock += 1
                if block not in cache:
                    while (
                        len(cache) * block_size + block_size
                    ) > capacity_tokens and lru:
                        evict_block = min(lru, key=lru.get)
                        del cache[evict_block]
                        del lru[evict_block]

                    if (
                        len(cache) * block_size + block_size
                    ) <= capacity_tokens:
                        cache[block] = True
                        lru[block] = clock
                else:
                    lru[block] = clock
    else:
        raise ValueError(f"Unknown cache_type: {cache_type}")

    return hit_tokens / total_tokens if total_tokens > 0 else 0.0
