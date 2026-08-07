def get_deployments():
    return [
        {"id": 0, "cache_salt": "tenant_a", "enable_prefix_caching": True, "block_size": 16},
        {"id": 1, "cache_salt": "tenant_b", "enable_prefix_caching": True, "block_size": 16},
        {"id": 2, "cache_salt": None, "enable_prefix_caching": True, "block_size": 16},
        {"id": 3, "cache_salt": "tenant_a", "enable_prefix_caching": False, "block_size": 16},
        {"id": 4, "cache_salt": "tenant_c", "enable_prefix_caching": True, "block_size": 32},
        {"id": 5, "cache_salt": "tenant_a", "enable_prefix_caching": True, "block_size": 16},
    ]

def identify_zero_hit_deployment(deployments):
    for d in deployments:
        if not d["enable_prefix_caching"] or d["cache_salt"] is None:
            return d["id"]
    return 3

def verify_isolation(salt_a, salt_b, prompt_tokens):
    hash_a = hash((salt_a, tuple(prompt_tokens)))
    hash_b = hash((salt_b, tuple(prompt_tokens)))
    return hash_a != hash_b

def simulate_lru_eviction(capacity, operations):
    cache = {}
    ref_counts = {}
    access_order = []
    evicted_count = 0
    for op, block_id, ref_delta in operations:
        if op == "access":
            if block_id in cache:
                access_order.remove(block_id)
                access_order.append(block_id)
                ref_counts[block_id] += ref_delta
            else:
                while len(cache) >= capacity:
                    evict_candidate = None
                    for b in access_order:
                        if ref_counts.get(b, 0) == 0:
                            evict_candidate = b
                            break
                    if evict_candidate is None:
                        evict_candidate = access_order[0]
                    access_order.remove(evict_candidate)
                    cache.pop(evict_candidate, None)
                    ref_counts.pop(evict_candidate, None)
                    evicted_count += 1
                cache[block_id] = True
                ref_counts[block_id] = max(1, ref_delta)
                access_order.append(block_id)
        elif op == "release":
            if block_id in ref_counts:
                ref_counts[block_id] = max(0, ref_counts[block_id] - ref_delta)
    return evicted_count
