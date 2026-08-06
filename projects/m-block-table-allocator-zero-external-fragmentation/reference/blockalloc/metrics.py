def measure_prefix_hit_rate(requests, block_size, cache_capacity_blocks):
    """Simulates prefix cache eviction and measures prefix hit rate on server workload."""
    cache = {}
    lru_counter = 0
    total_tokens = 0
    hit_tokens = 0

    for req in requests:
        tokens = req["tokens"]
        total_tokens += len(tokens)
        
        num_full_blocks = len(tokens) // block_size
        matched_tokens = 0
        
        for b_idx in range(num_full_blocks):
            block_tokens = tuple(tokens[b_idx * block_size : (b_idx + 1) * block_size])
            if block_tokens in cache:
                matched_tokens += block_size
                lru_counter += 1
                cache[block_tokens]["last_used"] = lru_counter
            else:
                break
        
        hit_tokens += matched_tokens

        for b_idx in range(num_full_blocks):
            block_tokens = tuple(tokens[b_idx * block_size : (b_idx + 1) * block_size])
            lru_counter += 1
            if block_tokens in cache:
                cache[block_tokens]["last_used"] = lru_counter
            else:
                if len(cache) >= cache_capacity_blocks:
                    lru_key = min(cache.keys(), key=lambda k: cache[k]["last_used"])
                    del cache[lru_key]
                cache[block_tokens] = {"last_used": lru_counter}

    return hit_tokens / total_tokens if total_tokens > 0 else 0.0
