def simulate_session_routing(sessions, num_replicas, use_affinity):
    replica_caches = [set() for _ in range(num_replicas)]
    total_tokens = 0
    hits = 0
    turn_counter = 0
    for session_id, turns in sessions:
        for turn in turns:
            tokens = turn.get("prefix_tokens", [])
            total_tokens += len(tokens)
            if use_affinity:
                assigned_replica = hash(session_id) % num_replicas
            else:
                assigned_replica = turn_counter % num_replicas
            turn_counter += 1
            cache = replica_caches[assigned_replica]
            for tok in tokens:
                if tok in cache:
                    hits += 1
                else:
                    cache.add(tok)
    hit_rate = (hits / float(total_tokens)) if total_tokens > 0 else 0.0
    return {
        "hit_rate": hit_rate,
        "hits": hits,
        "total_tokens": total_tokens,
    }
